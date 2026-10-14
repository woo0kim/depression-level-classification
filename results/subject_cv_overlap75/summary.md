# Results: subject-wise 4-fold CV

Segments: 1-s windows, 75% overlap, 1197 per subject. Mean ± std over folds. Positive class = more severe group. Subject-level prediction = mean of the subject's 1197 segment probabilities.

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

## Segment-level

| Task | Model | ACC | Bal. ACC | F1 | Precision | Recall | Specificity | AUC |
|---|---|---|---|---|---|---|---|---|
| None vs (Mild + Severe) | EEGNet | 0.571 ± 0.076 | 0.561 ± 0.068 | 0.486 ± 0.044 | 0.539 ± 0.112 | 0.453 ± 0.022 | 0.669 ± 0.150 | 0.535 ± 0.090 |
| None vs (Mild + Severe) | ShallowFBCSPNet | 0.550 ± 0.070 | 0.542 ± 0.059 | 0.472 ± 0.028 | 0.512 ± 0.093 | 0.453 ± 0.054 | 0.632 ± 0.165 | 0.539 ± 0.075 |
| None vs (Mild + Severe) | Deep4Net | 0.567 ± 0.059 | 0.552 ± 0.059 | 0.452 ± 0.088 | 0.528 ± 0.099 | 0.411 ± 0.112 | 0.694 ± 0.123 | 0.552 ± 0.092 |
| (None + Mild) vs Severe | EEGNet | 0.563 ± 0.096 | 0.451 ± 0.053 | 0.182 ± 0.051 | 0.137 ± 0.049 | 0.282 ± 0.067 | 0.621 ± 0.124 | 0.430 ± 0.097 |
| (None + Mild) vs Severe | ShallowFBCSPNet | 0.545 ± 0.071 | 0.461 ± 0.086 | 0.200 ± 0.104 | 0.144 ± 0.077 | 0.336 ± 0.162 | 0.586 ± 0.083 | 0.425 ± 0.134 |
| (None + Mild) vs Severe | Deep4Net | 0.535 ± 0.130 | 0.464 ± 0.072 | 0.174 ± 0.142 | 0.117 ± 0.091 | 0.362 ± 0.326 | 0.566 ± 0.213 | 0.425 ± 0.137 |
| Mild vs Severe | EEGNet | 0.564 ± 0.050 | 0.582 ± 0.064 | 0.514 ± 0.072 | 0.457 ± 0.073 | 0.626 ± 0.202 | 0.538 ± 0.141 | 0.618 ± 0.115 |
| Mild vs Severe | ShallowFBCSPNet | 0.555 ± 0.022 | 0.545 ± 0.031 | 0.448 ± 0.088 | 0.427 ± 0.055 | 0.495 ± 0.177 | 0.596 ± 0.128 | 0.564 ± 0.047 |
| Mild vs Severe | Deep4Net | 0.513 ± 0.092 | 0.550 ± 0.058 | 0.513 ± 0.104 | 0.419 ± 0.071 | 0.705 ± 0.262 | 0.395 ± 0.243 | 0.507 ± 0.099 |

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
