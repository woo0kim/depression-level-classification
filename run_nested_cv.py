"""Nested subject-wise cross-validation (4 outer x 3 inner folds) with hyper-parameter selection.

    python run_nested_cv.py --labels "../20260616_Demographic_grouped (1).csv"

Protocol, per task and model (all splits are by subject and stratified on the task label):
  1. Outer 4-fold split. The outer test subjects are never used for model selection.
  2. Inner 3-fold split of the outer-training subjects (2/3 train, 1/3 validation, rotated).
     Every grid config is trained on each inner split with early stopping on the class-weighted
     validation loss (patience 10, max 50 epochs); best epoch = lowest validation loss.
  3. Selected config = highest mean subject-level validation balanced accuracy (at the best
     epochs) over the 3 inner folds; ties -> lower mean validation loss -> grid order.
     Final number of epochs = median of the selected config's 3 inner best epochs.
  4. Retrain from scratch on all outer-training subjects for that many epochs (same LR schedule
     as the inner runs, no validation data).
  5. Predict the outer test subjects; subject probability = mean of its segment probabilities.
  6. Pool the outer-test predictions (each subject is tested exactly once) and compute the
     metrics once on the pooled set.

Jobs can be split across processes / GPUs with --tasks and --models. Each (task, model) job
writes to <out>/runs/<task>__<model>/ and resumes where it stopped. --summarize-only pools
whatever has finished and writes summary.md / summary.csv.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from deplevel.data import (TASKS, attach_labels, flat, load_cache, load_labels, segment, standardize,
                           task_subset)
from deplevel.models import MODEL_NAMES, build_model, complexity
from deplevel.train import binary_metrics, fit, pick_device, predict_proba, set_seed

METRICS = ("acc", "bal_acc", "precision", "recall", "specificity", "f1", "f1_macro", "auc")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default="../20260616_Demographic_grouped (1).csv")
    p.add_argument("--cache", default="cache/f3f4_mid300s_128hz.npz")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    p.add_argument("--outer-folds", type=int, default=4)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--overlap", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 1e-4])
    p.add_argument("--dropouts", nargs="+", type=float, default=[0.25, 0.5])
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/nested_cv_overlap75")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--quiet", action="store_true", help="don't print per-epoch logs")
    return p.parse_args()


def config_grid(args):
    return [{"config": f"lr={lr:g}, dropout={d:g}", "lr": lr, "drop_prob": d}
            for lr in args.lrs for d in args.dropouts]


def make_splits(y, args):
    """[(outer_train, outer_test, [(inner_train, inner_val), ...]), ...] as indices into y."""
    outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    splits = []
    for k, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y)):
        inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + k)
        splits.append((tr, te, [(tr[a], tr[b]) for a, b in inner.split(np.zeros(len(tr)), y[tr])]))
    return splits


def _read_csv(path):
    return pd.read_csv(path).to_dict("records") if os.path.exists(path) else []


def run_job(task, model_name, x_task, y, s_task, args, device, log):
    job = os.path.join(args.out, "runs", f"{task}__{model_name}")
    os.makedirs(job, exist_ok=True)
    paths = {k: os.path.join(job, f) for k, f in [("inner", "inner_cv.csv"), ("sel", "selection.csv"),
                                                  ("pred", "predictions.csv"), ("hist", "history.json")]}
    inner_rows, sel_rows, pred_rows = _read_csv(paths["inner"]), _read_csv(paths["sel"]), _read_csv(paths["pred"])
    histories = json.load(open(paths["hist"])) if os.path.exists(paths["hist"]) else {}
    done_inner = {(r["outer_fold"], r["config"], r["inner_fold"]) for r in inner_rows}
    done_outer = {r["outer_fold"] for r in pred_rows}
    sel_rows = [r for r in sel_rows if r["outer_fold"] in done_outer]

    def save():
        pd.DataFrame(inner_rows).to_csv(paths["inner"], index=False)
        pd.DataFrame(sel_rows).to_csv(paths["sel"], index=False)
        pd.DataFrame(pred_rows).to_csv(paths["pred"], index=False)
        with open(paths["hist"], "w") as f:
            json.dump(histories, f)

    n_seg = x_task.shape[1]
    grid = config_grid(args)
    train_kw = dict(batch_size=args.batch_size, weight_decay=args.weight_decay, log=log)
    for k, (otr, ote, inner_splits) in enumerate(make_splits(y, args), 1):
        if k in done_outer:
            print(f"[{task} {model_name}] outer fold {k}: already done, skipping", flush=True)
            continue

        # Steps 2-3: inner CV over the grid, using the outer-training subjects only.
        for c in grid:
            for j, (itr, iva) in enumerate(inner_splits, 1):
                if (k, c["config"], j) in done_inner:
                    continue
                t0 = time.time()
                xtr, xva = standardize(x_task[itr], x_task[iva])
                (xtr, ytr), (xva, yva) = flat(xtr, y[itr]), flat(xva, y[iva])
                seed = args.seed + 100 * k + j
                set_seed(seed)
                model = build_model(model_name, n_chans=x_task.shape[2], n_times=x_task.shape[3],
                                    drop_prob=c["drop_prob"])
                _, best_epoch, hist = fit(model, xtr, ytr, xva, yva, device, epochs=args.max_epochs,
                                          patience=args.patience, lr=c["lr"], seed=seed,
                                          val_subject_size=n_seg, **train_kw)
                b = hist[best_epoch - 1]
                inner_rows.append({"outer_fold": k, "config": c["config"], "lr": c["lr"],
                                   "drop_prob": c["drop_prob"], "inner_fold": j, "best_epoch": best_epoch,
                                   "epochs_run": len(hist), "val_loss": b["val_loss"],
                                   "val_sub_bal_acc": b["val_sub_bal_acc"], "val_sub_acc": b["val_sub_acc"],
                                   "val_sub_auc": b["val_sub_auc"], "n_train": len(itr), "n_val": len(iva),
                                   "seconds": round(time.time() - t0, 1)})
                histories[f"outer{k}/{c['config']}/inner{j}"] = hist
                save()
                print(f"[{task} {model_name}] outer {k} | {c['config']} | inner {j}: best epoch {best_epoch}, "
                      f"val loss {b['val_loss']:.4f}, val subject bal-acc {b['val_sub_bal_acc']:.3f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)

        inner = pd.DataFrame([r for r in inner_rows if r["outer_fold"] == k])
        agg = (inner.groupby("config", sort=False)
               .agg(mean_val_sub_bal_acc=("val_sub_bal_acc", "mean"), mean_val_loss=("val_loss", "mean"),
                    best_epochs=("best_epoch", list))
               .reindex([c["config"] for c in grid]).reset_index())
        agg["grid_order"] = range(len(agg))
        chosen = agg.sort_values(["mean_val_sub_bal_acc", "mean_val_loss", "grid_order"],
                                 ascending=[False, True, True]).iloc[0]
        c = next(g for g in grid if g["config"] == chosen["config"])
        n_epochs = int(np.median(chosen["best_epochs"]))

        # Step 4: retrain from scratch on all outer-training subjects for n_epochs.
        t0 = time.time()
        xtr, xte = standardize(x_task[otr], x_task[ote])
        (xtr, ytr), (xte, yte) = flat(xtr, y[otr]), flat(xte, y[ote])
        seed = args.seed + 100 * k
        set_seed(seed)
        model = build_model(model_name, n_chans=x_task.shape[2], n_times=x_task.shape[3],
                            drop_prob=c["drop_prob"])
        model, _, hist = fit(model, xtr, ytr, None, None, device, epochs=n_epochs,
                             schedule_epochs=args.max_epochs, lr=c["lr"], seed=seed, **train_kw)
        histories[f"outer{k}/final"] = hist

        # Step 5: subject-level prediction on the untouched outer test subjects.
        p_seg = predict_proba(model, xte, device)[:, 1]
        p_sub = p_seg.reshape(len(ote), -1).mean(axis=1)
        seg_correct = ((p_seg >= 0.5).reshape(len(ote), -1) == y[ote][:, None]).sum(axis=1)
        for s, yt, ps, sc in zip(s_task[ote], y[ote], p_sub, seg_correct):
            pred_rows.append({"task": task, "model": model_name, "outer_fold": k, "subject": s,
                              "y_true": int(yt), "prob_pos": float(ps), "y_pred": int(ps >= 0.5),
                              "n_segments": n_seg, "segments_correct": int(sc)})
        sel_rows.append({"task": task, "model": model_name, "outer_fold": k, "config": c["config"],
                         "lr": c["lr"], "drop_prob": c["drop_prob"],
                         "mean_val_sub_bal_acc": chosen["mean_val_sub_bal_acc"],
                         "mean_val_loss": chosen["mean_val_loss"],
                         "inner_best_epochs": "/".join(map(str, chosen["best_epochs"])),
                         "final_epochs": n_epochs, "n_outer_train": len(otr), "n_outer_test": len(ote)})
        save()
        fold_acc = np.mean((p_sub >= 0.5) == y[ote])
        print(f"[{task} {model_name}] outer {k}: selected {c['config']}, {n_epochs} epochs "
              f"(inner best {chosen['best_epochs']}) -> outer-test subject ACC {fold_acc:.3f} "
              f"[final fit {time.time() - t0:.0f}s]", flush=True)


def pooled_metrics(pred):
    y, p = pred["y_true"].to_numpy(), pred["prob_pos"].to_numpy()
    m = binary_metrics(y, p)
    tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]).ravel()
    fold_acc = pred.groupby("outer_fold").apply(lambda d: np.mean(d.y_pred == d.y_true))
    # Segment level, pooled over all outer-test segments.
    corr = pred.groupby("y_true")["segments_correct"].sum()
    tot = pred.groupby("y_true")["n_segments"].sum()
    return {**m, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n_subjects": len(pred),
            "fold_acc_mean": fold_acc.mean(), "fold_acc_std": fold_acc.std(),
            "seg_acc": corr.sum() / tot.sum(), "seg_bal_acc": float(np.mean(corr / tot))}


def summarize(args):
    preds = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.out, "runs", "*", "predictions.csv")))]
    sels = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.out, "runs", "*", "selection.csv")))]
    if not preds:
        print("no finished outer folds yet")
        return
    pred, sel = pd.concat(preds, ignore_index=True), pd.concat(sels, ignore_index=True)
    pred.to_csv(os.path.join(args.out, "pooled_predictions.csv"), index=False)
    sel.to_csv(os.path.join(args.out, "selected_hyperparameters.csv"), index=False)

    rows, incomplete = [], []
    for task in TASKS:
        for model in MODEL_NAMES:
            d = pred[(pred.task == task) & (pred.model == model)]
            if d.empty:
                continue
            if d.outer_fold.nunique() < args.outer_folds:
                incomplete.append(f"{task} / {model}: {d.outer_fold.nunique()}/{args.outer_folds} outer folds")
                continue
            assert d.subject.is_unique, f"{task}/{model}: a subject was tested more than once"
            rows.append({"task": task, "model": model, **pooled_metrics(d)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    n_seg = int(pred.n_segments.iloc[0])
    L = ["# Results: nested subject-wise CV", "",
         f"Outer {args.outer_folds}-fold × inner {args.inner_folds}-fold CV, split by subject. "
         f"1-s windows, {args.overlap:.0%} overlap ({n_seg:,} per subject). Hyper-parameter grid: "
         f"lr {{{', '.join(f'{v:g}' for v in args.lrs)}}} × dropout {{{', '.join(f'{v:g}' for v in args.dropouts)}}}. "
         "Every subject is tested exactly once; metrics are computed once on the pooled outer-test predictions. "
         "Subject probability = mean of its segment probabilities, threshold 0.5. "
         "Positive class = more severe group.", ""]
    if incomplete:
        L += ["**Incomplete (not shown):** " + "; ".join(incomplete), ""]
    L += ["## Pooled subject-level results", "",
          "| Task | Model | ACC | Bal. ACC | F1 | Precision | Recall | Specificity | AUC | TN / FP / FN / TP |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {TASKS[r.task]['description']} | {r.model} | " +
                 " | ".join(f"{r[m]:.3f}" for m in ("acc", "bal_acc", "f1", "precision", "recall", "specificity", "auc")) +
                 f" | {r.tn} / {r.fp} / {r.fn} / {r.tp} |")
    L += ["", "## Pooled vs. fold-averaged accuracy", "",
          "Pooled ACC = correctly classified subjects / all subjects. "
          "Fold-mean ACC weights each outer fold equally (shown for reference only).", "",
          "| Task | Model | Subjects | Pooled ACC | Fold-mean ACC ± std | Pooled segment ACC | Pooled segment Bal. ACC |",
          "|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {TASKS[r.task]['description']} | {r.model} | {r.n_subjects} | {r.acc:.3f} | "
                 f"{r.fold_acc_mean:.3f} ± {r.fold_acc_std:.3f} | {r.seg_acc:.3f} | {r.seg_bal_acc:.3f} |")
    L += ["", "## Selected hyper-parameters per outer fold", "",
          "Config chosen by mean inner-validation subject-level balanced accuracy; epochs = median of the "
          "3 inner best epochs (lowest validation loss).", "",
          "| Task | Model | " + " | ".join(f"Outer fold {k}" for k in range(1, args.outer_folds + 1)) + " |",
          "|---|---|" + "---|" * args.outer_folds]
    for (task, model), d in sel.groupby(["task", "model"], sort=False):
        d = d.set_index("outer_fold")
        cells = [f"lr {d.loc[k, 'lr']:g}, p {d.loc[k, 'drop_prob']:g}, {d.loc[k, 'final_epochs']} ep"
                 if k in d.index else "–" for k in range(1, args.outer_folds + 1)]
        L.append(f"| {TASKS[task]['description']} | {model} | " + " | ".join(cells) + " |")
    L += ["", "## Majority-class baseline", "", "| Task | Subjects (neg / pos) | Majority ACC |", "|---|---|---|"]
    for task in res.task.unique():
        y = pred[(pred.task == task)].drop_duplicates("subject").y_true
        L.append(f"| {TASKS[task]['description']} | {(y == 0).sum()} / {(y == 1).sum()} | "
                 f"{max(y.mean(), 1 - y.mean()):.3f} |")
    L += ["", "## Model complexity (input 2 × 128)", "",
          "| Model | Params | Size (KB, fp32) | FLOPs / segment |", "|---|---|---|---|"]
    for m in res.model.unique():
        cx = complexity(build_model(m))
        L.append(f"| {m} | {cx['params']:,} | {cx['size_kb']:.1f} | {cx['mflops']:.3f} M |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.summarize_only:
        summarize(args)
        return
    device = pick_device(args.device)
    log = (lambda *a, **k: None) if args.quiet else print

    labels = load_labels(args.labels)
    data, unlabeled, _ = attach_labels(load_cache(args.cache), labels)
    segs = segment(data["signals"], overlap=args.overlap)
    print(f"{len(data['subjects'])} labelled subjects, segments {segs.shape}, device {device}", flush=True)
    if unlabeled:
        print(f"  recordings without label (excluded): {unlabeled}", flush=True)

    with open(os.path.join(args.out, f"config_{'_'.join(args.tasks)}_{'_'.join(args.models)}.json"), "w") as f:
        json.dump({**vars(args), "device": str(device), "grid": config_grid(args)}, f, indent=2)
    for task in args.tasks:
        idx, y = task_subset(data["severity"], task)
        x_task, s_task = segs[idx], data["subjects"][idx]
        # One file per task, so parallel jobs never overwrite each other.
        with open(os.path.join(args.out, f"splits_{task}.json"), "w") as f:
            json.dump([{"outer_test": s_task[te].tolist(),
                        "inner": [{"train": s_task[i].tolist(), "val": s_task[v].tolist()} for i, v in inner]}
                       for _, te, inner in make_splits(y, args)], f, indent=1)
        for model_name in args.models:
            print(f"\n=== {task} ({TASKS[task]['description']}) / {model_name}: "
                  f"{(y == 0).sum()} vs {(y == 1).sum()} subjects", flush=True)
            run_job(task, model_name, x_task, y, s_task, args, device, log)

    summarize(args)


if __name__ == "__main__":
    main()
