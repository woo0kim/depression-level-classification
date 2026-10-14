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

## Segment-level

| Task | Model | ACC | Bal. ACC | F1 | Precision | Recall | Specificity | AUC |
|---|---|---|---|---|---|---|---|---|
| None vs (Mild + Severe) | EEGNet | 0.553 ± 0.080 | 0.530 ± 0.069 | 0.299 ± 0.235 | 0.553 ± 0.165 | 0.279 ± 0.253 | 0.780 ± 0.279 | 0.530 ± 0.088 |
| None vs (Mild + Severe) | ShallowFBCSPNet | 0.545 ± 0.065 | 0.542 ± 0.060 | 0.493 ± 0.063 | 0.493 ± 0.076 | 0.501 ± 0.077 | 0.583 ± 0.115 | 0.531 ± 0.071 |
| None vs (Mild + Severe) | Deep4Net | 0.572 ± 0.076 | 0.555 ± 0.073 | 0.440 ± 0.083 | 0.542 ± 0.119 | 0.382 ± 0.088 | 0.728 ± 0.140 | 0.535 ± 0.102 |
| (None + Mild) vs Severe | EEGNet | 0.661 ± 0.204 | 0.478 ± 0.069 | 0.119 ± 0.139 | 0.160 ± 0.115 | 0.209 ± 0.259 | 0.748 ± 0.286 | 0.467 ± 0.117 |
| (None + Mild) vs Severe | ShallowFBCSPNet | 0.582 ± 0.059 | 0.467 ± 0.081 | 0.194 ± 0.105 | 0.145 ± 0.079 | 0.297 ± 0.154 | 0.638 ± 0.068 | 0.440 ± 0.122 |
| (None + Mild) vs Severe | Deep4Net | 0.625 ± 0.102 | 0.451 ± 0.083 | 0.119 ± 0.105 | 0.111 ± 0.068 | 0.183 ± 0.213 | 0.719 ± 0.160 | 0.430 ± 0.115 |
| Mild vs Severe | EEGNet | 0.471 ± 0.132 | 0.512 ± 0.021 | 0.436 ± 0.247 | 0.465 ± 0.143 | 0.725 ± 0.460 | 0.299 ± 0.467 | 0.547 ± 0.078 |
| Mild vs Severe | ShallowFBCSPNet | 0.580 ± 0.030 | 0.561 ± 0.037 | 0.434 ± 0.110 | 0.457 ± 0.080 | 0.456 ± 0.213 | 0.666 ± 0.167 | 0.610 ± 0.034 |
| Mild vs Severe | Deep4Net | 0.531 ± 0.108 | 0.558 ± 0.059 | 0.511 ± 0.087 | 0.442 ± 0.079 | 0.669 ± 0.255 | 0.446 ± 0.279 | 0.529 ± 0.093 |

## Majority-class baseline (subject-level ACC)

| Task | Subjects (neg / pos) | Majority ACC |
|---|---|---|
| None vs (Mild + Severe) | 69 / 55 | 0.556 |
| (None + Mild) vs Severe | 103 / 21 | 0.831 |
| Mild vs Severe | 34 / 21 | 0.618 |

## Model complexity (input 2 × 128)

| Model | Params | Size (KB, fp32) | FLOPs / segment |
|---|---|---|---|
| EEGNet | 1,266 | 5.3 | 0.306 M |
| ShallowFBCSPNet | 4,482 | 17.8 | 0.420 M |
| Deep4Net | 135,002 | 530.3 | 3.641 M |
