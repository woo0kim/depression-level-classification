# Results: nested subject-wise CV (1-s windows, 75 % overlap)

Outer 4-fold × inner 3-fold cross-validation, split by subject. Hyper-parameters and the number of epochs are chosen inside each outer fold using only its training subjects; the final model is retrained on all outer-training subjects and tested once on the held-out outer-test subjects. Each subject receives exactly one outer-test prediction (mean of its 1,197 segment probabilities, threshold 0.5), and all metrics are computed once on these pooled predictions. Positive class = more severe group.

## Pooled subject-level results

| Task | Model | ACC | Bal. ACC | F1 | Precision | Recall | Specificity | AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|---|
| None vs (Mild + Severe) | EEGNet | 0.581 | 0.573 | 0.519 | 0.528 | 0.509 | 0.638 | 0.576 | 44 / 25 / 27 / 28 |
| None vs (Mild + Severe) | ShallowFBCSPNet | 0.605 | 0.595 | 0.533 | 0.560 | 0.509 | 0.681 | 0.582 | 47 / 22 / 27 / 28 |
| None vs (Mild + Severe) | Deep4Net | 0.548 | 0.533 | 0.440 | 0.489 | 0.400 | 0.667 | 0.530 | 46 / 23 / 33 / 22 |
| (None + Mild) vs Severe | EEGNet | 0.597 | 0.378 | 0.038 | 0.032 | 0.048 | 0.709 | 0.385 | 73 / 30 / 20 / 1 |
| (None + Mild) vs Severe | ShallowFBCSPNet | 0.621 | 0.469 | 0.175 | 0.139 | 0.238 | 0.699 | 0.421 | 72 / 31 / 16 / 5 |
| (None + Mild) vs Severe | Deep4Net | 0.556 | 0.354 | 0.035 | 0.028 | 0.048 | 0.660 | 0.408 | 68 / 35 / 20 / 1 |
| Mild vs Severe | EEGNet | 0.600 | 0.613 | 0.560 | 0.483 | 0.667 | 0.559 | 0.581 | 19 / 15 / 7 / 14 |
| Mild vs Severe | ShallowFBCSPNet | 0.509 | 0.485 | 0.372 | 0.364 | 0.381 | 0.588 | 0.436 | 20 / 14 / 13 / 8 |
| Mild vs Severe | Deep4Net | 0.545 | 0.560 | 0.510 | 0.433 | 0.619 | 0.500 | 0.480 | 17 / 17 / 8 / 13 |

Majority-class baseline ACC (always predicting the larger group): None vs (Mild + Severe) 0.556 (69 / 124), (None + Mild) vs Severe 0.831 (103 / 124), Mild vs Severe 0.618 (34 / 55). Balanced ACC and AUC are 0.5 at chance level.

## Selected hyper-parameters per outer fold

Each cell: learning rate, p = dropout, ep = final number of epochs (median of the 3 inner best epochs).

| Task | Model | Outer fold 1 | Outer fold 2 | Outer fold 3 | Outer fold 4 |
|---|---|---|---|---|---|
| None vs (Mild + Severe) | EEGNet | 1e-4, p 0.5, 1 ep | 1e-3, p 0.5, 2 ep | 1e-3, p 0.5, 1 ep | 1e-3, p 0.5, 4 ep |
| None vs (Mild + Severe) | ShallowFBCSPNet | 1e-4, p 0.5, 1 ep | 1e-3, p 0.5, 1 ep | 1e-4, p 0.5, 10 ep | 1e-3, p 0.25, 3 ep |
| None vs (Mild + Severe) | Deep4Net | 1e-3, p 0.5, 1 ep | 1e-3, p 0.5, 2 ep | 1e-3, p 0.5, 1 ep | 1e-4, p 0.25, 2 ep |
| (None + Mild) vs Severe | EEGNet | 1e-4, p 0.25, 1 ep | 1e-3, p 0.5, 16 ep | 1e-3, p 0.5, 1 ep | 1e-3, p 0.5, 2 ep |
| (None + Mild) vs Severe | ShallowFBCSPNet | 1e-3, p 0.5, 1 ep | 1e-3, p 0.5, 2 ep | 1e-3, p 0.5, 1 ep | 1e-4, p 0.5, 5 ep |
| (None + Mild) vs Severe | Deep4Net | 1e-4, p 0.25, 2 ep | 1e-3, p 0.25, 2 ep | 1e-4, p 0.25, 1 ep | 1e-4, p 0.25, 2 ep |
| Mild vs Severe | EEGNet | 1e-4, p 0.25, 3 ep | 1e-3, p 0.25, 1 ep | 1e-3, p 0.25, 1 ep | 1e-3, p 0.5, 1 ep |
| Mild vs Severe | ShallowFBCSPNet | 1e-3, p 0.5, 1 ep | 1e-3, p 0.25, 1 ep | 1e-4, p 0.5, 1 ep | 1e-3, p 0.5, 1 ep |
| Mild vs Severe | Deep4Net | 1e-4, p 0.25, 1 ep | 1e-4, p 0.5, 1 ep | 1e-3, p 0.25, 1 ep | 1e-4, p 0.5, 5 ep |

## Experimental settings

All three models use the same protocol, grid and folds.

| Category | Setting | Value |
|---|---|---|
| Input | Channels / window | F3, F4 / middle 300 s of each recording |
| | Sampling rate | 128 Hz (from 1000 Hz, polyphase resampling) |
| | Segments | 1-s windows, **75 % overlap** (0.25-s step), applied uniformly to all subjects: 1,197 per subject (148,428 in total) |
| | Normalisation | per-channel z-score, mean/std from the training subjects of each (inner or outer) training set |
| Outer CV | Scheme | stratified 4-fold, **split by subject**; outer-test subjects are never used for model selection |
| | Subjects (train / test) | 93 / 31 (None vs (Mild + Severe), (None + Mild) vs Severe); 41–42 / 13–14 (Mild vs Severe) |
| Inner CV | Scheme | stratified 3-fold on the outer-training subjects, split by subject (2/3 train, 1/3 validation, rotated) |
| | Subjects (train / validation) | 62 / 31; 27–28 / 13–14 (Mild vs Severe) |
| | Hyper-parameter grid | learning rate {1e-3, 1e-4} × dropout {0.25, 0.5} (4 configs) |
| | Best epoch per inner fold | lowest class-weighted validation loss; max 50 epochs, early stopping patience 10 |
| | Config selection | highest mean subject-level validation balanced accuracy over the 3 inner folds (ties: lower mean validation loss) |
| | Final number of epochs | median of the selected config's 3 inner best epochs (rule fixed in advance) |
| Final model | Training | retrained from scratch on all outer-training subjects for the final number of epochs, no validation data |
| Training | Loss | cross-entropy weighted by inverse class frequency (training set) |
| | Optimiser / LR schedule | AdamW (lr from grid, weight decay 1e-4) / cosine annealing over 50 epochs (same trajectory in inner and final training) |
| | Batch size | 128 segments |
| | Random seeds | 42 (outer split), 42 + outer fold (inner split), 42 + 100 · outer + inner (inner runs), 42 + 100 · outer (final model) |
| Evaluation | Subject level | mean of the subject's 1,197 segment probabilities, threshold 0.5 |
| | Pooling | one outer-test prediction per subject; metrics computed once on all subjects (124; 55 for Mild vs Severe) |
| | Metrics | ACC, balanced ACC, precision, recall (sensitivity), specificity, F1, ROC-AUC, confusion matrix |

Subjects are split into folds. Splitting by segment would put 1-s pieces of the same recording in both training and test sets. The model could then recognise the person instead of their depression level, which inflates accuracy. With 75 % overlap this would be even worse, because neighbouring windows share 96 of their 128 samples.

## Models

| Model | Configuration | Params | Size (fp32) | FLOPs / 1-s segment |
|---|---|---|---|---|
| EEGNet (`EEGNetv4`) | braindecode (EEGNet-8,2; kernel 64 = 0.5 s at 128 Hz); dropout from grid | 1,266 | 5.3 KB | 0.31 M |
| ShallowFBCSPNet | braindecode defaults; dropout from grid | 4,482 | 17.8 KB | 0.42 M |
| Deep4Net | kernels 5, pools 2/2 (see below); dropout from grid | 135,002 | 530 KB | 3.64 M |

The default Deep4Net (temporal kernel 10, max-pool 3/3 in four blocks, designed for 250 Hz / 4-s trials) cannot process a 128-sample input: the time axis shrinks below 1. Kernels and pools are scaled to roughly 128/250 of the original (kernel 5, pool 2 stride 2), giving time lengths 128 → 62 → 29 → 12 → 4. Filter counts and everything else are unchanged.

## Environment

Lab server with 8 × NVIDIA Quadro M6000 (12 GB); Python 3.11.7, PyTorch 2.1.2 (CUDA 12.1), braindecode 0.8. The 468 training runs (432 inner + 36 final) took 52 min, with the 9 task × model jobs running in parallel. cuDNN was set to deterministic mode, and a re-run on the same GPU type reproduced the results exactly.
