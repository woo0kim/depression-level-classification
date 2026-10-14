"""Subject-wise 4-fold cross-validation of EEGNet / ShallowFBCSPNet / Deep4Net on the
three depression-severity tasks.

    python run_cv.py --labels "../20260616_Demographic_grouped (1).csv"

Folds are split by subject (stratified on the task label), so all 300 segments of a
subject are in exactly one of train / validation / test. Within each training fold,
`--val-frac` of the training subjects are held out for early stopping. Every model
sees the same folds. Metrics are reported per 1-s segment and per subject (mean
segment probability).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from deplevel.data import TASKS, attach_labels, load_cache, load_labels, segment, task_subset
from deplevel.models import MODEL_NAMES, build_model, complexity
from deplevel.train import binary_metrics, fit, pick_device, predict_proba, set_seed

METRICS = ("acc", "bal_acc", "precision", "recall", "specificity", "f1", "f1_macro", "auc")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default="../20260616_Demographic_grouped (1).csv")
    p.add_argument("--cache", default="cache/f3f4_mid300s_128hz.npz")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/subject_cv")
    p.add_argument("--quiet", action="store_true", help="don't print per-epoch logs")
    return p.parse_args()


def standardize(train, *others):
    """Channel-wise z-score with statistics from the training subjects only."""
    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)
    return [(a - mean) / std for a in (train, *others)]


def flat(x, y):
    """(n_sub, n_seg, ch, t), (n_sub,) -> (n_sub*n_seg, ch, t), (n_sub*n_seg,)."""
    return x.reshape(-1, *x.shape[2:]), np.repeat(y, x.shape[1])


def summarize(fold_df, cx_df, baseline, out):
    agg = fold_df.groupby(["task", "model", "level"], sort=False)[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(out, "summary.csv"), index=False)

    lines = ["# Results: subject-wise 4-fold CV", ""]
    lines.append("Mean ± std over folds. Positive class = more severe group. "
                 "Subject-level prediction = mean of the subject's 300 segment probabilities.")
    lines.append("")
    for level in ("subject", "segment"):
        lines += [f"## {level.capitalize()}-level", "",
                  "| Task | Model | ACC | Bal. ACC | F1 | Precision | Recall | Specificity | AUC |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in agg[agg.level == level].iterrows():
            cells = [f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}"
                     for m in ("acc", "bal_acc", "f1", "precision", "recall", "specificity", "auc")]
            lines.append(f"| {TASKS[r.task]['description']} | {r.model} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Majority-class baseline (subject-level ACC)", "",
              "| Task | Subjects (neg / pos) | Majority ACC |", "|---|---|---|"]
    for task, b in baseline.items():
        lines.append(f"| {TASKS[task]['description']} | {b['n_neg']} / {b['n_pos']} | {b['majority_acc']:.3f} |")
    lines += ["", "## Model complexity (input 2 × 128)", "",
              "| Model | Params | Size (KB, fp32) | FLOPs / segment |", "|---|---|---|---|"]
    for _, r in cx_df.iterrows():
        lines.append(f"| {r.model} | {r.params:,} | {r.size_kb:.1f} | {r.mflops:.3f} M |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    log = (lambda *a, **k: None) if args.quiet else print

    labels = load_labels(args.labels)
    data, unlabeled, no_eeg = attach_labels(load_cache(args.cache), labels)
    segs = segment(data["signals"])  # (n_subjects, 300, 2, 128)
    subjects, severity = data["subjects"], data["severity"]
    print(f"{len(subjects)} labelled subjects, segments {segs.shape}, device {device}")
    print(f"  severity counts (None/Mild/Severe): {np.bincount(severity, minlength=3).tolist()}")
    if unlabeled:
        print(f"  recordings without label (excluded): {unlabeled}")
    if no_eeg:
        print(f"  labelled subjects without recording: {no_eeg}")

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "device": str(device), "n_subjects": len(subjects),
                   "excluded_recordings": unlabeled}, f, indent=2)

    cx_df = pd.DataFrame([{"model": m, **complexity(build_model(m))} for m in args.models])
    cx_df.to_csv(os.path.join(args.out, "complexity.csv"), index=False)

    rows, preds, histories, splits, baseline = [], [], {}, {}, {}
    for task in args.tasks:
        idx, y = task_subset(severity, task)
        x_task, s_task = segs[idx], subjects[idx]
        baseline[task] = {"n_neg": int((y == 0).sum()), "n_pos": int((y == 1).sum()),
                          "majority_acc": max(np.mean(y), 1 - np.mean(y))}
        print(f"\n=== {task}: {TASKS[task]['description']}  "
              f"({baseline[task]['n_neg']} vs {baseline[task]['n_pos']} subjects)")

        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits[task] = []
        for fold, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            tr, va = train_test_split(tr_all, test_size=args.val_frac, stratify=y[tr_all],
                                      random_state=args.seed + fold)
            splits[task].append({"train": s_task[tr].tolist(), "val": s_task[va].tolist(),
                                 "test": s_task[te].tolist()})
            x_tr, x_va, x_te = standardize(x_task[tr], x_task[va], x_task[te])
            xtr_f, ytr_f = flat(x_tr, y[tr])
            xva_f, yva_f = flat(x_va, y[va])
            xte_f, yte_f = flat(x_te, y[te])

            for model_name in args.models:
                t0 = time.time()
                print(f"  [{task}] fold {fold + 1}/{args.folds} {model_name}: "
                      f"train {len(tr)} / val {len(va)} / test {len(te)} subjects")
                set_seed(args.seed + fold)
                model = build_model(model_name, n_chans=segs.shape[2], n_times=segs.shape[3])
                model, best_epoch, hist = fit(
                    model, xtr_f, ytr_f, xva_f, yva_f, device, epochs=args.epochs,
                    patience=args.patience, batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, seed=args.seed + fold, log=log)
                histories[f"{task}/{model_name}/fold{fold + 1}"] = hist

                p_seg = predict_proba(model, xte_f, device)[:, 1]
                p_sub = p_seg.reshape(len(te), -1).mean(axis=1)
                common = {"task": task, "model": model_name, "fold": fold + 1, "best_epoch": best_epoch,
                          "n_test_subjects": len(te)}
                seg_m = binary_metrics(yte_f, p_seg)
                sub_m = binary_metrics(y[te], p_sub)
                rows.append({**common, "level": "segment", **seg_m})
                rows.append({**common, "level": "subject", **sub_m})
                seg_correct = (p_seg >= 0.5).reshape(len(te), -1) == y[te][:, None]
                for s, yt, ps, sc in zip(s_task[te], y[te], p_sub, seg_correct.mean(axis=1)):
                    preds.append({**common, "subject": s, "y_true": int(yt), "prob_pos": float(ps),
                                  "y_pred": int(ps >= 0.5), "segment_acc": float(sc)})
                print(f"    -> best epoch {best_epoch}, segment ACC {seg_m['acc']:.3f}, "
                      f"subject ACC {sub_m['acc']:.3f} (bal {sub_m['bal_acc']:.3f}) "
                      f"[{time.time() - t0:.0f}s]")

                # Write after every run so partial results survive an interruption.
                fold_df = pd.DataFrame(rows)
                fold_df.to_csv(os.path.join(args.out, "fold_metrics.csv"), index=False)
                pd.DataFrame(preds).to_csv(os.path.join(args.out, "subject_predictions.csv"), index=False)
                with open(os.path.join(args.out, "history.json"), "w") as f:
                    json.dump(histories, f)
                with open(os.path.join(args.out, "splits.json"), "w") as f:
                    json.dump(splits, f, indent=1)

    print("\n" + summarize(pd.DataFrame(rows), cx_df, baseline, args.out))


if __name__ == "__main__":
    main()
