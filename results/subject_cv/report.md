# Results: subject-wise 4-fold CV

Mean ± std over folds. Positive class = more severe group. Subject-level prediction = mean of the subject's 300 segment probabilities.

## Subject-level

| Task | Model | ACC | Bal. ACC | F1 | Precision | Recall | Specificity | AUC |
|---|---|---|---|---|---|---|---|---|
| None vs (Mild + Severe) | EEGNet | 0.556 ± 0.107 | 0.532 ± 0.092 | 0.279 ± 0.255 | 0.530 ± 0.438 | 0.260 ± 0.264 | 0.804 ± 0.319 | 0.575 ± 0.121 |
| None vs (Mild + Severe) | ShallowFBCSPNet | 0.589 ± 0.107 | 0.584 ± 0.102 | 0.534 ± 0.101 | 0.550 ± 0.135 | 0.527 ± 0.089 | 0.641 ± 0.155 | 0.563 ± 0.116 |
| None vs (Mild + Severe) | Deep4Net | 0.597 ± 0.085 | 0.579 ± 0.079 | 0.467 ± 0.087 | 0.596 ± 0.146 | 0.401 ± 0.098 | 0.757 ± 0.174 | 0.560 ± 0.124 |
| (None + Mild) vs Severe | EEGNet | 0.669 ± 0.206 | 0.466 ± 0.153 | 0.095 ± 0.190 | 0.067 ± 0.133 | 0.167 ± 0.333 | 0.765 ± 0.272 | 0.453 ± 0.168 |
| (None + Mild) vs Severe | ShallowFBCSPNet | 0.597 ± 0.056 | 0.447 ± 0.108 | 0.158 ± 0.137 | 0.122 ± 0.102 | 0.225 ± 0.206 | 0.670 ± 0.052 | 0.427 ± 0.146 |
| (None + Mild) vs Severe | Deep4Net | 0.637 ± 0.107 | 0.445 ± 0.101 | 0.102 ± 0.129 | 0.078 ± 0.097 | 0.150 ± 0.191 | 0.740 ± 0.151 | 0.428 ± 0.148 |
| Mild vs Severe | EEGNet | 0.453 ± 0.130 | 0.500 ± 0.000 | 0.420 ± 0.282 | 0.293 ± 0.197 | 0.750 ± 0.500 | 0.250 ± 0.500 | 0.579 ± 0.105 |
| Mild vs Severe | ShallowFBCSPNet | 0.618 ± 0.067 | 0.592 ± 0.071 | 0.393 ± 0.264 | 0.468 ± 0.410 | 0.433 ± 0.346 | 0.750 ± 0.292 | 0.637 ± 0.075 |
| Mild vs Severe | Deep4Net | 0.560 ± 0.179 | 0.584 ± 0.153 | 0.535 ± 0.172 | 0.484 ± 0.179 | 0.700 ± 0.346 | 0.469 ± 0.324 | 0.573 ± 0.121 |


## Experimental settings

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

Subjects are split into folds. Splitting by segment would put 1-s pieces
of the same recording in both training and test sets. The model could then recognise
the person instead of their depression level, which inflates accuracy.



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

