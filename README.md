# Depression Level Classification from Frontal Resting-State EEG

Binary depression-severity classification from two frontal EEG channels (F3, F4) with
EEGNet, ShallowFBCSPNet and Deep4Net (braindecode), evaluated with subject-wise 4-fold
cross-validation.

## Data

| | |
|---|---|
| Recordings | `Baseline_pre-RECT/*_pre_cleaned.mat` (resting state, pre-treatment; 1–100 Hz filtered, average referenced, ICA-cleaned, 1000 Hz, 31 ch) |
| Labels | `20260616_Demographic_grouped (1).csv`, rows whose `Subject ID` ends in `_pre`, column `HAMD17_group` |
| Severity groups | None/Remission (HAMD-17 0–7): **69**, Mild–Moderate (8–18): **34**, Severe (≥19): **21**; total **124** |

Recordings are matched to labels on the sheet's `file name` column (= `Clean_EEG.subjectID`),
not on `Subject ID`, because the two prefixes differ for one subject
(`TRD026_pre` → `111C022_CYC_20230621_resting`). Four recordings have no row in the
sheet and are excluded: `K015`, `nTRD029`, `TRD039`, `TRD049`.

## Preprocessing (`preprocess.py`)

1. Read F3 and F4 **by channel name** (their column index differs between files).
2. Take the **middle 300 s** of the recording.
3. Downsample 1000 → **128 Hz** (polyphase, `scipy.signal.resample_poly`; 1 s of
   extra signal on each side is resampled and trimmed to avoid filter edge effects).
4. Cut into non-overlapping **1-s segments** → 300 segments × (2 ch × 128 samples) per subject.

Within each CV fold, signals are z-scored per channel with the training subjects' mean/std.

## Tasks

| Key | Task | Subjects (neg / pos) |
|---|---|---|
| `c1_none_vs_dep` | None vs (Mild + Severe) | 69 / 55 |
| `c2_nonemild_vs_severe` | (None + Mild) vs Severe | 103 / 21 |
| `c3_mild_vs_severe` | Mild vs Severe | 34 / 21 |

The positive class (1) is always the more severe group.

## Models (`deplevel/models.py`)

| Model | Configuration | Params | Size (fp32) | FLOPs / 1-s segment |
|---|---|---|---|---|
| EEGNet (`EEGNetv4`) | braindecode defaults (EEGNet-8,2; kernel 64 = 0.5 s at 128 Hz) | 1,266 | 5.3 KB | 0.31 M |
| ShallowFBCSPNet | braindecode defaults | 4,482 | 17.8 KB | 0.42 M |
| Deep4Net | kernels 5, pools 2/2 (see below) | 135,002 | 530 KB | 3.64 M |

The default Deep4Net (temporal kernel 10, max-pool 3/3 in four blocks, designed for
250 Hz / 4-s trials) cannot process a 128-sample input: the time axis shrinks below 1.
Kernels and pools are scaled to roughly 128/250 of the original (kernel 5, pool 2 stride 2),
giving time lengths 128 → 62 → 29 → 12 → 4. Filter counts, dropout and everything else
are unchanged.

FLOPs are counted with `torch.utils.flop_counter.FlopCounterMode` (conv + linear layers,
1 multiply-add = 2 FLOPs).

## Experimental settings (`run_cv.py`)

All three models use the same settings and the same folds. No hyper-parameters were
tuned; the values below were fixed before the run.

| Category | Setting | Value |
|---|---|---|
| Input | Channels / window | F3, F4 / middle 300 s of each recording |
| | Sampling rate | 128 Hz (from 1000 Hz, polyphase resampling) |
| | Segments | 1 s, non-overlapping, 300 per subject (37,200 in total) |
| | Normalisation | per-channel z-score, mean/std from the training subjects of each fold |
| Cross-validation | Scheme | stratified 4-fold, **split by subject** (all 300 segments of a subject are in one fold) |
| | Validation set | 20 % of the training subjects (stratified), used only for early stopping |
| | Subjects per fold (train / val / test) | Task 1–2: 74 / 19 / 31; Task 3: 32–33 / 9 / 13–14 |
| | Random seed | 42 (fold split); 42 + fold index (inner split, weight init, batch order) |
| | Repetitions | 1 (a single 4-fold CV run) |
| Training | Loss | cross-entropy weighted by inverse class frequency (training set) |
| | Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| | LR schedule | cosine annealing over 50 epochs |
| | Batch size | 128 segments |
| | Epochs | max 50; early stopping on validation loss, patience 10; best-validation weights restored |
| Evaluation | Segment level | each 1-s segment classified independently |
| | Subject level | mean of the subject's 300 segment probabilities, threshold 0.5 |
| | Metrics | ACC, balanced ACC, precision, recall (sensitivity), specificity, F1, macro-F1, ROC-AUC; positive class = more severe group; mean ± std over the 4 test folds |
| | Baseline | majority-class ACC (always predicting the larger group) |
| Complexity | Params / size | all parameters and buffers, fp32 |
| | FLOPs | `torch.utils.flop_counter.FlopCounterMode` for one 2 × 128 input; conv + linear layers, 1 multiply-add = 2 FLOPs |

Subjects, not segments, are split into folds. Splitting by segment would put 1-s pieces
of the same recording in both training and test sets. The model could then recognise
the person instead of their depression level, which inflates accuracy.

Not included (intentionally, to follow the specified pipeline): hyper-parameter search,
data augmentation, notch filtering or extra artifact rejection beyond the provided ICA
cleaning, and the use of age or sex as inputs or covariates.

## Environment

| | |
|---|---|
| Hardware | Apple M4 (10-core CPU, 10-core GPU), 16 GB unified memory |
| OS | macOS 27.0.1 |
| Accelerator | Apple GPU via PyTorch **MPS** backend (no NVIDIA GPU / CUDA) |
| Software | Python 3.9.6, PyTorch 2.8.0, braindecode 0.8, MNE 1.8.0, NumPy 2.0.2, SciPy 1.13.1, scikit-learn 1.6.1, pandas 2.3.2, h5py 3.14.0 |
| Runtime | preprocessing 43 s (CPU); all 36 training runs (3 tasks × 3 models × 4 folds) 15 min on MPS. CPU-only (`--device cpu`) is estimated at ≈ 50 min from per-epoch timings. |

**A GPU is not required.** The models are small (1.3 k–135 k parameters, ≤ 3.6 MFLOPs
per segment) and the whole dataset is 37,200 segments of 2 × 128 samples (≈ 38 MB),
so everything fits in memory and runs on a CPU. The GPU only shortens the run. Per
epoch on Task 1 it is ≈ 6× faster for EEGNet (1.9 s vs 11.8 s) and ≈ 3× for Deep4Net
(3.0 s vs 8.8 s). ShallowFBCSPNet is faster on the CPU (0.85 s vs 2.2 s). `--device auto`
(the default) picks CUDA, then MPS, then CPU.

**Reproducibility.** Re-running on the same device gives identical results; this was checked
for Mild vs Severe / ShallowFBCSPNet on MPS. Running on a different device (CPU, CUDA)
gives slightly different floating-point results. Early stopping can then select a
different epoch, so per-fold numbers shift a little. In that check, accuracy was unchanged,
but AUC changed in 2 of 4 folds. Reported numbers come from the MPS run.

## Usage

```bash
pip install -r requirements.txt

# 1) preprocess the .mat files once (→ cache/f3f4_mid300s_128hz.npz, ~40 MB)
python preprocess.py --data-dir ../Baseline_pre-RECT

# 2) run all tasks × models (≈ 15 min on Apple M4 GPU/MPS, ≈ 50 min on CPU)
python run_cv.py --labels "../20260616_Demographic_grouped (1).csv" --out results/subject_cv

# subsets / options
python run_cv.py --tasks c3_mild_vs_severe --models EEGNet --epochs 100 --device cpu
```

Outputs in `--out`:

| File | Content |
|---|---|
| `summary.md` / `summary.csv` | mean ± std over folds per task × model × level, majority baseline, complexity |
| `fold_metrics.csv` | per-fold metrics (segment and subject level) and best epoch |
| `subject_predictions.csv` | per-subject test predictions (mean probability, prediction, segment ACC) |
| `splits.json` | train / val / test subject IDs per task and fold |
| `history.json` | per-epoch train/val loss and val balanced ACC |
| `complexity.csv` | params, size, FLOPs |
| `config.json`, `train.log` | run configuration and console log |

## Results

Subject-wise 4-fold CV, seed 42, mean ± std over folds. Full tables (precision, recall,
specificity, F1, macro-F1) are in [`results/subject_cv/summary.md`](results/subject_cv/summary.md).

**Majority ACC** is the accuracy of a trivial classifier that always predicts the larger
class, for example "None+Mild" for every subject in Task 2 (103 / 124 = 0.831). A model
must beat this to show that it learned something from the EEG. With imbalanced classes,
accuracy alone can hide a model that ignores the minority class. Balanced ACC (mean of
sensitivity and specificity, 0.5 = chance) and AUC (0.5 = chance) do not depend on the
class ratio and should be read alongside it.

| Task | Model | Segment ACC | Subject ACC | Subject Bal. ACC | Subject AUC | Majority ACC |
|---|---|---|---|---|---|---|
| None vs (Mild + Severe) | EEGNet | 0.553 ± 0.080 | 0.556 ± 0.107 | 0.532 ± 0.092 | 0.575 ± 0.121 | 0.556 |
| | ShallowFBCSPNet | 0.545 ± 0.065 | 0.589 ± 0.107 | 0.584 ± 0.102 | 0.563 ± 0.116 | |
| | Deep4Net | 0.572 ± 0.076 | 0.597 ± 0.085 | 0.579 ± 0.079 | 0.560 ± 0.124 | |
| (None + Mild) vs Severe | EEGNet | 0.661 ± 0.204 | 0.669 ± 0.206 | 0.466 ± 0.153 | 0.453 ± 0.168 | 0.831 |
| | ShallowFBCSPNet | 0.582 ± 0.059 | 0.597 ± 0.056 | 0.447 ± 0.108 | 0.427 ± 0.146 | |
| | Deep4Net | 0.625 ± 0.102 | 0.637 ± 0.107 | 0.445 ± 0.101 | 0.428 ± 0.148 | |
| Mild vs Severe | EEGNet | 0.471 ± 0.132 | 0.453 ± 0.130 | 0.500 ± 0.000 | 0.579 ± 0.105 | 0.618 |
| | ShallowFBCSPNet | 0.580 ± 0.030 | 0.618 ± 0.067 | 0.592 ± 0.071 | 0.637 ± 0.075 | |
| | Deep4Net | 0.531 ± 0.108 | 0.560 ± 0.179 | 0.584 ± 0.153 | 0.573 ± 0.121 | |

Observations:

- Across subjects, F3/F4 1-s segments carry little class information for these
  networks. Most results are at or below the majority-class baseline. The best cases are
  weakly above chance in balanced ACC / AUC: Shallow and Deep4 on Task 1, and
  ShallowFBCSPNet on Task 3 (AUC 0.64).
- On (None + Mild) vs Severe, plain ACC is misleading (majority baseline 0.83). Balanced
  ACC and AUC are below 0.5, so no model detects the 21 severe subjects.
- The best validation epoch is usually 1–6. Validation loss on held-out subjects bottoms
  out almost immediately, after which the models only fit the training subjects.
  EEGNet on Mild vs Severe never left its initialisation: its outputs stay near 0.5 and it
  predicts a single class in each fold, hence Bal. ACC 0.500 ± 0.000.
- With 31 test subjects per fold (14 for Mild vs Severe), fold-to-fold std is about
  0.1. Differences between models are within that noise.
- Data note: `TRD067_pre` (Severe) has an F3 standard deviation of 35 µV against 4 µV
  on F4 and about 6 µV typical, which suggests a residual artifact on F3.
