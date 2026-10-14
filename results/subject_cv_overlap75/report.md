# Results: subject-wise 4-fold CV (1-s windows, 75 % overlap)

Mean ± std over folds. Positive class = more severe group. Subject-level prediction = mean of the subject's 1,197 segment probabilities.

## Subject-level

| Task | Model | ACC | Bal. ACC | F1 | Precision | Recall | Specificity | AUC |
|---|---|---|---|---|---|---|---|---|
| None vs (Mild + Severe) | EEGNet | 0.605 ± 0.113 | 0.591 ± 0.098 | 0.501 ± 0.066 | 0.657 ± 0.262 | 0.438 ± 0.075 | 0.743 ± 0.248 | 0.569 ± 0.128 |
| None vs (Mild + Severe) | ShallowFBCSPNet | 0.524 ± 0.072 | 0.513 ± 0.054 | 0.405 ± 0.085 | 0.485 ± 0.085 | 0.385 ± 0.159 | 0.642 ± 0.234 | 0.567 ± 0.113 |
| None vs (Mild + Severe) | Deep4Net | 0.540 ± 0.089 | 0.525 ± 0.085 | 0.406 ± 0.115 | 0.500 ± 0.138 | 0.365 ± 0.137 | 0.684 ± 0.172 | 0.564 ± 0.116 |
| (None + Mild) vs Severe | EEGNet | 0.605 ± 0.103 | 0.441 ± 0.061 | 0.145 ± 0.027 | 0.122 ± 0.039 | 0.192 ± 0.017 | 0.690 ± 0.131 | 0.409 ± 0.133 |
| (None + Mild) vs Severe | ShallowFBCSPNet | 0.556 ± 0.103 | 0.459 ± 0.139 | 0.198 ± 0.150 | 0.145 ± 0.111 | 0.317 ± 0.233 | 0.602 ± 0.104 | 0.387 ± 0.169 |
| (None + Mild) vs Severe | Deep4Net | 0.532 ± 0.180 | 0.468 ± 0.052 | 0.168 ± 0.132 | 0.110 ± 0.084 | 0.375 ± 0.350 | 0.562 ± 0.279 | 0.409 ± 0.171 |
| Mild vs Severe | EEGNet | 0.563 ± 0.061 | 0.584 ± 0.093 | 0.507 ± 0.114 | 0.451 ± 0.061 | 0.633 ± 0.275 | 0.535 ± 0.146 | 0.638 ± 0.175 |
| Mild vs Severe | ShallowFBCSPNet | 0.599 ± 0.078 | 0.591 ± 0.103 | 0.469 ± 0.177 | 0.484 ± 0.184 | 0.525 ± 0.340 | 0.656 ± 0.232 | 0.555 ± 0.087 |
| Mild vs Severe | Deep4Net | 0.508 ± 0.094 | 0.543 ± 0.106 | 0.497 ± 0.166 | 0.399 ± 0.100 | 0.700 ± 0.346 | 0.385 ± 0.198 | 0.550 ± 0.149 |

## Comparison with non-overlapping windows

Subject-level means over the 4 folds. Same folds, seeds and training settings; only the window step differs (1 s vs 0.25 s).

| Task | Model | ACC: no overlap | ACC: 75 % overlap | Bal. ACC: no overlap | Bal. ACC: 75 % overlap | AUC: no overlap | AUC: 75 % overlap |
|---|---|---|---|---|---|---|---|
| None vs (Mild + Severe) | EEGNet | 0.556 | 0.605 | 0.532 | 0.591 | 0.575 | 0.569 |
| None vs (Mild + Severe) | ShallowFBCSPNet | 0.589 | 0.524 | 0.584 | 0.513 | 0.563 | 0.567 |
| None vs (Mild + Severe) | Deep4Net | 0.597 | 0.540 | 0.579 | 0.525 | 0.560 | 0.564 |
| (None + Mild) vs Severe | EEGNet | 0.669 | 0.605 | 0.466 | 0.441 | 0.453 | 0.409 |
| (None + Mild) vs Severe | ShallowFBCSPNet | 0.597 | 0.556 | 0.447 | 0.459 | 0.427 | 0.387 |
| (None + Mild) vs Severe | Deep4Net | 0.637 | 0.532 | 0.445 | 0.468 | 0.428 | 0.409 |
| Mild vs Severe | EEGNet | 0.453 | 0.563 | 0.500 | 0.584 | 0.579 | 0.638 |
| Mild vs Severe | ShallowFBCSPNet | 0.618 | 0.599 | 0.592 | 0.591 | 0.637 | 0.555 |
| Mild vs Severe | Deep4Net | 0.560 | 0.508 | 0.584 | 0.543 | 0.573 | 0.550 |


## Experimental settings

All three models use the same settings and the same folds. No hyper-parameters were
tuned; the values below were fixed before the run.

| Category | Setting | Value |
|---|---|---|
| Input | Channels / window | F3, F4 / middle 300 s of each recording |
| | Sampling rate | 128 Hz (from 1000 Hz, polyphase resampling) |
| | Segments | 1 s windows, **75 % overlap** (0.25-s step), 1,197 per subject (148,428 in total) |
| | Normalisation | per-channel z-score, mean/std from the training subjects of each fold |
| Cross-validation | Scheme | stratified 4-fold, **split by subject** (all segments of a subject are in one fold) |
| | Validation set | 20 % of the training subjects (stratified), used only for early stopping |
| | Subjects per fold (train / val / test) | Task 1–2: 74 / 19 / 31; Task 3: 32–33 / 9 / 13–14 |
| | Random seed | 42 (fold split); 42 + fold index (inner split, weight init, batch order) |
| | Repetitions | 1 (a single 4-fold CV run) |
| Training | Loss | cross-entropy weighted by inverse class frequency (training set) |
| | Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| | LR schedule | cosine annealing over 50 epochs |
| | Batch size | 128 segments (≈ 4× more steps per epoch than without overlap) |
| | Epochs | max 50; early stopping on validation loss, patience 10; best-validation weights restored |
| Evaluation | Segment level | each 1-s segment classified independently |
| | Subject level | mean of the subject's 1,197 segment probabilities, threshold 0.5 |
| | Metrics | ACC, balanced ACC, precision, recall (sensitivity), specificity, F1, macro-F1, ROC-AUC; positive class = more severe group; mean ± std over the 4 test folds |
| | Baseline | majority-class ACC (always predicting the larger group) |
| Complexity | Params / size | all parameters and buffers, fp32 |
| | FLOPs | `torch.utils.flop_counter.FlopCounterMode` for one 2 × 128 input; conv + linear layers, 1 multiply-add = 2 FLOPs |

Subjects are split into folds. Splitting by segment would put 1-s pieces
of the same recording in both training and test sets. The model could then recognise
the person instead of their depression level, which inflates accuracy. With 75 % overlap this would be even worse, because neighbouring windows share 96 of their 128 samples. The overlap is applied to training, validation and test windows; since every subject is in exactly one fold, overlapping windows never cross the train/test boundary.



## Models

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

