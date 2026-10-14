"""Loading, preprocessing and labelling of the resting-state EEG recordings.

Pipeline (per recording):
    Clean_EEG.data (1000 Hz, 31 ch) -> F3, F4 -> middle 300 s -> 128 Hz -> 1-s segments
"""
import glob
import os
from fractions import Fraction

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

CHANNELS = ("F3", "F4")
TARGET_SFREQ = 128
DURATION_S = 300
SEGMENT_S = 1
# Extra signal kept on each side of the 300-s window while resampling so the
# anti-aliasing filter has no edge transients inside the window.
PAD_S = 1

CLASS_NAMES = ("None", "Mild", "Severe")
NONE, MILD, SEVERE = 0, 1, 2

# Each task maps the 3-level severity label to a binary target (1 = more severe
# group). Severity levels not listed are excluded from the task.
TASKS = {
    "c1_none_vs_dep": {
        "description": "None vs (Mild + Severe)",
        "mapping": {NONE: 0, MILD: 1, SEVERE: 1},
        "class_names": ("None", "Mild+Severe"),
    },
    "c2_nonemild_vs_severe": {
        "description": "(None + Mild) vs Severe",
        "mapping": {NONE: 0, MILD: 0, SEVERE: 1},
        "class_names": ("None+Mild", "Severe"),
    },
    "c3_mild_vs_severe": {
        "description": "Mild vs Severe",
        "mapping": {MILD: 0, SEVERE: 1},
        "class_names": ("Mild", "Severe"),
    },
}


def _mat_str(ds):
    return "".join(map(chr, np.asarray(ds).flatten()))


def load_recording(path, channels=CHANNELS):
    """Return (data[n_channels, n_samples], sfreq, subjectID) from a cleaned .mat file."""
    with h5py.File(path, "r") as f:
        eeg = f["Clean_EEG"]
        labels = [_mat_str(f[ref]).upper() for ref in np.asarray(eeg["chanlabels"]).flatten()]
        # Channel order differs between files, so channels are looked up by name.
        idx = [labels.index(ch.upper()) for ch in channels]
        sfreq = float(np.asarray(eeg["srate"]).item())
        # MATLAB v7.3 stores the (channels x samples) array transposed.
        data = np.stack([eeg["data"][:, i] for i in idx])
        sid = _mat_str(eeg["subjectID"])
    return data.astype(np.float64), sfreq, sid


def extract_middle(data, sfreq, duration_s=DURATION_S, target_sfreq=TARGET_SFREQ, pad_s=PAD_S):
    """Middle `duration_s` seconds of `data`, resampled to `target_sfreq`."""
    n_win = int(round(duration_s * sfreq))
    if data.shape[1] < n_win:
        raise ValueError(f"recording is {data.shape[1] / sfreq:.1f} s, need {duration_s} s")
    start = (data.shape[1] - n_win) // 2
    pad = min(int(round(pad_s * sfreq)), start, data.shape[1] - start - n_win)
    chunk = data[:, start - pad:start + n_win + pad]

    up, down = _ratio(target_sfreq, sfreq)
    res = resample_poly(chunk, up, down, axis=1)
    pad_out = int(round(pad * target_sfreq / sfreq))
    n_out = int(round(duration_s * target_sfreq))
    return res[:, pad_out:pad_out + n_out].astype(np.float32)


def _ratio(target, source):
    fr = Fraction(int(round(target)), int(round(source)))
    return fr.numerator, fr.denominator


def build_cache(data_dir, cache_path, pattern="*_pre_cleaned.mat"):
    """Preprocess every matching recording and save continuous 128-Hz signals."""
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} in {data_dir}")
    signals, recordings, names = [], [], []
    for i, path in enumerate(files, 1):
        data, sfreq, rec = load_recording(path)
        signals.append(extract_middle(data, sfreq))
        recordings.append(rec)
        names.append(os.path.basename(path))
        print(f"[{i:3d}/{len(files)}] {names[-1]}  ({data.shape[1] / sfreq:.1f} s)", flush=True)
    if len(set(recordings)) != len(recordings):
        raise ValueError("duplicate recording names in data directory")
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(
        cache_path,
        signals=np.stack(signals),
        recordings=np.array(recordings),
        files=np.array(names),
        channels=np.array(CHANNELS),
        sfreq=TARGET_SFREQ,
    )
    return cache_path


def load_cache(cache_path):
    with np.load(cache_path) as z:
        return {k: z[k] for k in z.files}


def segment(signals, sfreq=TARGET_SFREQ, segment_s=SEGMENT_S, overlap=0.0):
    """(n_subjects, n_ch, n_times) -> (n_subjects, n_segments, n_ch, seg_len).

    Windows of `segment_s` seconds with fractional `overlap` between neighbours
    (0 = non-overlapping, 0.75 = 1-s windows every 0.25 s).
    """
    seg_len = int(round(segment_s * sfreq))
    step = int(round(seg_len * (1 - overlap)))
    if not 0 < step <= seg_len:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    # Returned as a strided view (no copy). Keep it that way: the memory layout sets the summation
    # order of the float32 mean/std in standardize(), so a contiguous copy changes results slightly.
    x = np.lib.stride_tricks.sliding_window_view(signals, seg_len, axis=2)[:, :, ::step]
    return x.transpose(0, 2, 1, 3)


def parse_severity(value):
    """Map a HAMD17_group entry ('None/Remission (0-7)', 'Mild-Moderate (8-18)',
    'Severe (>=19)') to NONE / MILD / SEVERE."""
    s = str(value).strip().lower()
    if "severe" in s:
        return SEVERE
    if "mild" in s:
        return MILD
    if "none" in s or "remission" in s:
        return NONE
    raise ValueError(f"unrecognised severity label: {value!r}")


def load_labels(path, condition="pre", id_col="Subject ID", file_col="file name",
                label_col="HAMD17_group"):
    """Read the demographic sheet; returns a DataFrame indexed by recording name
    (the sheet's `file name`, e.g. 'HC001_HDH_20230601_RECT') with columns
    subject (e.g. 'HC001_pre') and severity.

    Recordings are joined on `file name` rather than `Subject ID` because the two
    do not always share a prefix (TRD026_pre -> 111C022_CYC_20230621_resting).
    """
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df[id_col].astype(str).str.strip().str.endswith(f"_{condition}")]
    out = pd.DataFrame({
        "subject": df[id_col].astype(str).str.strip().values,
        "severity": [parse_severity(v) for v in df[label_col]],
    }, index=df[file_col].astype(str).str.strip().str.replace(r"\.mat$", "", regex=True))
    if out.index.duplicated().any():
        raise ValueError(f"duplicate file names in label sheet: {list(out.index[out.index.duplicated()])}")
    return out


def attach_labels(cache, labels):
    """Keep the cached recordings that have a label.

    Returns dict(signals, subjects, recordings, severity), the recordings without a
    label and the labelled subjects without a recording.
    """
    recs = [str(r) for r in cache["recordings"]]
    keep = [i for i, r in enumerate(recs) if r in labels.index]
    unlabeled = [r for r in recs if r not in labels.index]
    no_eeg = sorted(labels.loc[~labels.index.isin(recs), "subject"])
    rows = labels.loc[[recs[i] for i in keep]]
    data = {
        "signals": cache["signals"][keep],
        "subjects": rows["subject"].to_numpy(),
        "recordings": np.array([recs[i] for i in keep]),
        "severity": rows["severity"].to_numpy(),
    }
    return data, unlabeled, no_eeg


def task_subset(severity, task):
    """Indices and binary targets of the subjects that take part in `task`."""
    mapping = TASKS[task]["mapping"]
    idx = np.array([i for i, s in enumerate(severity) if s in mapping])
    y = np.array([mapping[severity[i]] for i in idx])
    return idx, y


def standardize(train, *others):
    """Channel-wise z-score with statistics from the training subjects only."""
    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)
    return [(a - mean) / std for a in (train, *others)]


def flat(x, y):
    """(n_sub, n_seg, ch, t), (n_sub,) -> (n_sub*n_seg, ch, t), (n_sub*n_seg,)."""
    return x.reshape(-1, *x.shape[2:]), np.repeat(y, x.shape[1])
