# Quickstart Guide

## Installation

```bash
pip install threshscore
```

## Core Concepts

threshscore answers a practical question: *does my classifier meet the operating constraints my use-case demands?*

For example, a medical screening model might need at least 90% sensitivity (low false-negative rate) while also maintaining 70% specificity. Standard metrics like F1 or accuracy don't capture this. threshscore uses a **centering-based formula** that produces a score of exactly 0.5 at your operating point (when metrics exactly meet their thresholds), > 0.5 when both are exceeded, and < 0.5 when either falls short.

## Step 1: Prepare Your Data

```python
import numpy as np

# Ground-truth labels (0 = negative, 1 = positive)
y_true  = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])

# Predicted probability scores from your model
y_score = np.array([0.1, 0.2, 0.85, 0.92, 0.71, 0.35, 0.63, 0.44, 0.78, 0.55])

# Hard predictions at a chosen operating threshold
y_pred  = (y_score >= 0.5).astype(int)
```

## Step 2: Compute a Constrained Score

Three entry points cover every workflow:

```python
import threshscore

# From binary predictions (no probability scores needed)
result = threshscore.score_from_predictions(
    y_true, y_pred,
    s_thr=0.85,   # need ≥ 85% sensitivity
    p_thr=0.60,   # need ≥ 60% specificity
)

# From probability scores (binarises at threshold=0.5 by default)
result = threshscore.score_from_scores(
    y_true, y_score,
    threshold=0.5,   # binarisation threshold
    s_thr=0.85,
    p_thr=0.60,
)

# From pre-computed sensitivity / specificity values
score = threshscore.score_from_metrics(
    sens=0.88, spec=0.72,
    s_thr=0.85,
    p_thr=0.60,
)

print(result.score)        # overall constrained score in [0, 1]
print(result.sensitivity)  # achieved sensitivity
print(result.specificity)  # achieved specificity
```

**Centering property:** `score_from_metrics(s_thr, p_thr, s_thr=s_thr, p_thr=p_thr)` always returns **exactly 0.5**.  This makes score thresholds interpretable: 0.5 is the boundary, > 0.5 means constraints are met.

## Step 3: Get All Standard Metrics

```python
m = threshscore.compute_metrics(y_true, y_pred, y_score)

print(m.sensitivity)  # = recall
print(m.specificity)
print(m.precision)
print(m.f1)
print(m.accuracy)
print(m.roc_auc)      # None if y_score not provided
print(m.pr_auc)
print(m.confusion_matrix)  # 2x2 numpy array [[TN, FP], [FN, TP]]
```

## Step 4: Sweep Thresholds

Find the best operating point across all thresholds:

```python
sweep = threshscore.sweep(y_true, y_score, n_thresholds=101)

# Best F1
best_f1 = sweep.best_by_f1()
print(f"Best F1 threshold: {best_f1.threshold:.2f}  F1={best_f1.f1:.3f}")

# Highest sensitivity that still meets a specificity floor
point = sweep.best_by_sensitivity_at_specificity(min_specificity=0.7)
if point:
    print(f"Threshold {point.threshold:.2f}: sens={point.sensitivity:.3f}, spec={point.specificity:.3f}")

# Highest specificity that still meets a sensitivity floor
point = sweep.best_by_specificity_at_sensitivity(min_sensitivity=0.85)
```

## Step 5: Visualise

All plot functions return `(Figure, Axes)` and accept an optional `ax` kwarg.

```python
import matplotlib.pyplot as plt

fig, ax = threshscore.plot.plot_roc(y_true, y_score)
plt.show()

fig, ax = threshscore.plot.plot_pr(y_true, y_score)
plt.show()

fig, ax = threshscore.plot.plot_threshold(y_true=y_true, y_score=y_score)
plt.show()

fig, ax = threshscore.plot.plot_confusion(m.confusion_matrix)
plt.show()

# Score heatmap: shows centering property visually (0.5 contour at operating point)
fig, ax = threshscore.plot.plot_heatmap(
    sensitivity_threshold=0.85,
    specificity_threshold=0.60,
    target_sensitivity=0.88,
    target_specificity=0.72,
)
plt.show()

fig, ax = threshscore.plot.plot_gate("arctan")
plt.show()
```

## Gate Functions

Each gate function defines the transition behaviour around the sensitivity and specificity thresholds. Score = 0.5 at the operating point for all of them.

| Name | Description | Key param | Behaviour |
|------|-------------|-----------|-----------|
| `arctan` | Smooth sigmoid-like shape; default | `k=25` | Sharp transition around thresholds |
| `sigmoid` | Classic logistic curve | `k=15` | Similar to arctan, slightly wider |
| `relu_clip` | Hard zero below threshold | (none) | Cliff: either metric below → gate clips to 0 |
| `linear_clip` | Proportional partial credit below threshold | (none) | Partial credit when close to thresholds |

```python
# Use sigmoid with higher steepness
result = threshscore.score_from_predictions(
    y_true, y_pred,
    gate="sigmoid",
    gate_params={"k": 30.0},
)

# List available gate functions
print(threshscore.gates.list_gates())

# Visualise gate curve
fig, ax = threshscore.plot.plot_gate("sigmoid", gate_params={"k": 30.0})
```

## Custom Gate Functions

```python
from threshscore.gates import BaseGate, register

class HardThreshold(BaseGate):
    @property
    def default_params(self):
        return {}

    def __call__(self, value, threshold, **params):
        return 1.0 if value >= threshold else 0.0

register("hard_threshold", HardThreshold())
result = threshscore.score_from_predictions(y_true, y_pred, gate="hard_threshold")
```

## Next Steps

- [API Reference](api_reference.md): complete function and class reference
- [Example notebook](examples/basic_usage.ipynb): end-to-end worked example
