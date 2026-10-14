"""Training / inference loop and evaluation metrics."""
import copy
import random
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Deterministic cuDNN convolutions, so CUDA runs are repeatable on the same GPU (no effect on CPU/MPS).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(y, n_classes=2):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return torch.tensor(len(y) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)


def _batches(n, batch_size, shuffle, generator=None):
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield order[i:i + batch_size]


@torch.no_grad()
def predict_proba(model, x, device, batch_size=1024):
    """Softmax probabilities for x (numpy, n x ch x times)."""
    model.eval()
    xt = torch.as_tensor(x, dtype=torch.float32)
    out = []
    for idx in _batches(len(xt), batch_size, shuffle=False):
        out.append(torch.softmax(model(xt[idx].to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


def fit(model, x_train, y_train, x_val, y_val, device, epochs=50, patience=10, batch_size=128,
        lr=1e-3, weight_decay=1e-4, seed=0, schedule_epochs=None, val_subject_size=None, log=print):
    """Train with class-weighted cross-entropy.

    With validation data, keep the weights with the lowest (class-weighted) validation loss and
    stop after `patience` epochs without improvement. With `x_val=None`, train for exactly
    `epochs` epochs and return the final weights.

    The cosine LR schedule spans `schedule_epochs` (default: `epochs`), so a fixed-length run can
    follow the same LR trajectory as an early-stopped run with a longer budget.
    `val_subject_size` (segments per validation subject, subject-major order) adds subject-level
    validation metrics (mean segment probability per subject) to the history.
    """
    model.to(device)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=device)
    has_val = x_val is not None
    if has_val:
        xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
        yv = torch.as_tensor(y_val, dtype=torch.long, device=device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(y_train).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_epochs or epochs)
    gen = torch.Generator().manual_seed(seed)

    best = {"loss": float("inf"), "epoch": 0, "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for idx in _batches(len(xt), batch_size, shuffle=True, generator=gen):
            idx = idx.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            n += len(idx)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": total / n}

        if not has_val:
            history.append(record)
            log(f"    epoch {epoch:3d}  train {total / n:.4f}  {time.time() - t0:.1f}s")
            continue

        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(xv[i]) for i in _batches(len(xv), 1024, shuffle=False)])
            val_loss = criterion(logits, yv).item()
            val_bacc = balanced_accuracy_score(y_val, logits.argmax(1).cpu().numpy())
        record.update(val_loss=val_loss, val_bal_acc=val_bacc)
        if val_subject_size:
            p_sub = torch.softmax(logits, 1)[:, 1].cpu().numpy().reshape(-1, val_subject_size).mean(1)
            sub = binary_metrics(np.asarray(y_val)[::val_subject_size], p_sub)
            record.update(val_sub_acc=sub["acc"], val_sub_bal_acc=sub["bal_acc"], val_sub_auc=sub["auc"])
        history.append(record)
        improved = val_loss < best["loss"]
        if improved:
            best = {"loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        sub_txt = f"  val_sub_bacc {record['val_sub_bal_acc']:.3f}" if val_subject_size else ""
        log(f"    epoch {epoch:3d}  train {total / n:.4f}  val {val_loss:.4f}  "
            f"val_bacc {val_bacc:.3f}{sub_txt}  {time.time() - t0:.1f}s{'  *' if improved else ''}")
        if epoch - best["epoch"] >= patience:
            break

    if not has_val:
        return model, epochs, history
    model.load_state_dict(best["state"])
    return model, best["epoch"], history


def binary_metrics(y_true, prob_pos, threshold=0.5):
    """Metrics for a binary task; positive class (1) is the more severe group."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(prob_pos) >= threshold).astype(int)
    m = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    m["auc"] = roc_auc_score(y_true, prob_pos) if len(np.unique(y_true)) == 2 else float("nan")
    return m
