# threshscore

[![CI](https://github.com/BiomedicalDataMining/threshscore/actions/workflows/ci.yml/badge.svg)](https://github.com/BiomedicalDataMining/threshscore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/threshscore.svg)](https://pypi.org/project/threshscore/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20562994-blue)](https://doi.org/10.5281/zenodo.20562994)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Threshold-aware binary classification scoring for Python.**

---

## The Problem

Standard metrics like F1-score or ROC AUC measure *overall* performance, but they ignore
deployment constraints. In medical screening, fraud detection, or any safety-critical
application, you need *both* high sensitivity (catch most cases) *and* high specificity
(limit false alarms) - simultaneously. A classifier with 95% sensitivity and 40%
specificity may look fine on F1 yet fail completely in production.

What is missing is a score that tells you directly: *does this classifier meet my
operating constraints, and by how much?*

---

## The Solution: Centering-based Scoring

threshscore introduces a score built around two operating constraints:

- `s_thr` - minimum sensitivity required (e.g. 0.80: detect at least 80% of positives)
- `p_thr` - minimum specificity required (e.g. 0.60: correctly reject at least 60% of negatives)

The score has a guaranteed **centering property**: it equals exactly **0.5 at the
target operating point**, rises above 0.5 when both constraints are exceeded, and falls
below 0.5 when either constraint is missed.

| Score | Meaning |
|-------|---------|
| **0.5** | classifier is exactly at the target operating point |
| **> 0.5** | both constraints are met with margin |
| **< 0.5** | at least one constraint is not met |

This makes the score directly interpretable and comparable across classifiers, tasks, and
threshold pairs - without needing to look at sensitivity and specificity separately.

<p align="center">
  <img src="https://raw.githubusercontent.com/BiomedicalDataMining/threshscore/main/docs/images/heatmap.png" width="60%"/>
</p>
<p align="center">
  <em>Score surface on the (sensitivity, specificity) plane with s_thr = 0.80, p_thr = 0.60.<br>
  Green = both constraints met (score > 0.5); Red = constraints not met (score < 0.5).<br>
  The score equals 0.5 exactly at the intersection of the two threshold lines.</em>
</p>

---

## Installation

```bash
pip install threshscore
```

Requires Python 3.9+, NumPy, scikit-learn, matplotlib.

---

## Quick Start

```python
import threshscore

# Score from pre-computed metrics (returns a float)
# Constraints: at least 80% sensitivity and 60% specificity

print(threshscore.score_from_metrics(0.80, 0.60, s_thr=0.80, p_thr=0.60))
# 0.5        <- exactly at the operating point

print(threshscore.score_from_metrics(0.90, 0.75, s_thr=0.80, p_thr=0.60))
# ~0.84      <- both constraints exceeded

print(threshscore.score_from_metrics(0.70, 0.50, s_thr=0.80, p_thr=0.60))
# ~0.33      <- constraints not met
```

```python
import numpy as np
import threshscore

y_true  = np.array([0, 0, 1, 1, 1, 0, 1, 0])
y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4])
y_pred  = (y_score >= 0.5).astype(int)

# Score from binary predictions (returns a ScoreResult object)
result = threshscore.score_from_predictions(
    y_true, y_pred, s_thr=0.80, p_thr=0.60
)
print(result.score)        # centering-based score
print(result.sensitivity)  # achieved sensitivity
print(result.specificity)  # achieved specificity

# Score from probability scores (binarises at threshold; includes ROC/PR AUC)
result = threshscore.score_from_scores(
    y_true, y_score, threshold=0.5, s_thr=0.80, p_thr=0.60
)

# All standard metrics at once
m = threshscore.compute_metrics(y_true, y_pred, y_score)
print(m.f1, m.roc_auc, m.pr_auc)

# Sweep all decision thresholds and find the best operating point
sweep = threshscore.sweep(y_true, y_score)
best  = sweep.best_by_sensitivity_at_specificity(min_specificity=0.60)

# Plots (all return (Figure, Axes))
fig, ax = threshscore.plot.plot_roc(y_true, y_score)
fig, ax = threshscore.plot.plot_pr(y_true, y_score)
fig, ax = threshscore.plot.plot_threshold(y_true=y_true, y_score=y_score)
fig, ax = threshscore.plot.plot_confusion(m.confusion_matrix)
fig, ax = threshscore.plot.plot_heatmap(
    sensitivity_threshold=0.80, specificity_threshold=0.60
)
fig, ax = threshscore.plot.plot_gate("arctan")
```

---

## How It Works

### Gate functions

The core building block is a **gate function**: a monotone S-curve anchored so that
`g(threshold, threshold) = 0.5`. It acts as a soft switch - values well above the
threshold saturate toward 1, values well below saturate toward 0.

Four gate functions are built in:

<p align="center">
  <img src="https://raw.githubusercontent.com/BiomedicalDataMining/threshscore/main/docs/images/gates.png" width="65%"/>
</p>
<p align="center">
  <em>All four built-in gate functions at threshold = 0.80. Every gate returns exactly 0.5
  at the threshold, regardless of the threshold value chosen.</em>
</p>

| Name | Shape | Key parameter |
|------|-------|---------------|
| `arctan` (default) | Smooth S-curve | `k=25` (steepness) |
| `sigmoid` | Smooth S-curve | `k=15` (steepness) |
| `relu_clip` | Hard zero below threshold, linear above | - |
| `linear_clip` | Linear everywhere, anchored at threshold | - |

### Sensitivity gate and specificity gate

One gate is applied per constraint. The sensitivity gate `g_s` maps the observed
sensitivity relative to `s_thr`; the specificity gate `g_p` maps the observed
specificity relative to `p_thr`. Each independently signals whether its constraint is
satisfied, and both are anchored at 0.5 at their respective thresholds.

<p align="center">
  <img src="https://raw.githubusercontent.com/BiomedicalDataMining/threshscore/main/docs/images/gate_sensitivity.png" width="48%"/>
  <img src="https://raw.githubusercontent.com/BiomedicalDataMining/threshscore/main/docs/images/gate_specificity.png" width="48%"/>
</p>
<p align="center">
  <em>Left: sensitivity gate g_s anchored at s_thr = 0.80.
  Right: specificity gate g_p anchored at p_thr = 0.60.
  The purple dot marks the centering point: g = 0.5 at the threshold.</em>
</p>

### Score formula

The final score combines the two gates with a balanced performance term:

```
q     = 0.5 * (sens + spec)                           # balanced accuracy
g_s   = 0.5 + (1/pi) * arctan(k * (sens - s_thr))    # sensitivity gate
g_p   = 0.5 + (1/pi) * arctan(k * (spec - p_thr))    # specificity gate
gate  = g_s * g_p                                      # soft-AND of both constraints
score = clip(0.5 + 0.5*(gate - 0.25) + 0.5*(q - q_thr), 0, 1)
```

where `q_thr = 0.5*(s_thr + p_thr)`.

The gate product `g_s * g_p` is a **soft-AND**: it rises above its baseline of 0.25 only
when *both* constraints are satisfied. This is the key distinction from additive penalty
approaches (like penalized balanced accuracy), which can mask a failing constraint behind
a strong result on the other.

**Centering proof:** when `sens = s_thr` and `spec = p_thr`, both gates equal 0.5, so
`gate = 0.25` and `q = q_thr`, giving `score = 0.5` exactly - for any choice of
threshold pair and any gate function.

---

## API Reference

| Entry point | Input | Returns |
|-------------|-------|---------|
| `score_from_metrics(sens, spec, ...)` | Pre-computed sensitivity and specificity | `float` |
| `score_from_predictions(y_true, y_pred, ...)` | Binary labels and predictions | `ScoreResult` |
| `score_from_scores(y_true, y_score, ...)` | Labels and probability scores | `ScoreResult` |
| `compute_metrics(y_true, y_pred, y_score)` | Labels, predictions, scores | `Metrics` |
| `sweep(y_true, y_score)` | Labels and probability scores | `SweepResult` |

`ScoreResult` exposes: `score`, `sensitivity`, `specificity`, `sensitivity_gate`,
`specificity_gate`, `gate_name`, `metrics`.

| Feature | Description |
|---------|-------------|
| **Centering property** | Score = 0.5 at the operating point by construction, for all built-in gates |
| **Gate functions** | `arctan` (default), `sigmoid`, `relu_clip`, `linear_clip`: pluggable and configurable |
| **Standard metrics** | Sensitivity, specificity, precision, recall, F1, accuracy, ROC AUC, PR AUC, confusion matrix |
| **Threshold sweep** | Evaluate every decision threshold; rank by F1, sensitivity, or specificity constraint |
| **6 plot types** | ROC, PR curve, threshold vs. metrics, confusion matrix, score heatmap, gate shape |
| **Custom gate functions** | Subclass `BaseGate` and register with one line |
| **Type-annotated** | Full `mypy --strict` compliance |
| **Well-tested** | 188 tests, property-based tests via Hypothesis |

---

## Gate Function Reference

```python
# Use a different built-in gate
result = threshscore.score_from_predictions(
    y_true, y_pred,
    gate="sigmoid",
    gate_params={"k": 20.0},
)

# Visualise all gate shapes
fig, ax = threshscore.plot.plot_gate(threshold=0.8)

# Define and register a custom gate
from threshscore.gates import BaseGate, register

class StepGate(BaseGate):
    @property
    def default_params(self):
        return {}

    def __call__(self, value: float, threshold: float, **params) -> float:
        return 1.0 if value >= threshold else 0.0

register("step", StepGate())
result = threshscore.score_from_predictions(y_true, y_pred, gate="step")
```

---

## Documentation

- [Quickstart guide](https://github.com/BiomedicalDataMining/threshscore/blob/main/docs/quickstart.md)
- [API reference](https://github.com/BiomedicalDataMining/threshscore/blob/main/docs/api_reference.md)
- [Example notebook](https://github.com/BiomedicalDataMining/threshscore/blob/main/docs/examples/basic_usage.ipynb)

## Contributing

threshscore is primarily a reproducible research tool accompanying an IEEE publication.
Bug reports and questions are welcome via
[GitHub Issues](https://github.com/BiomedicalDataMining/threshscore/issues).
For code contributions (new gate functions, fixes), see [CONTRIBUTING.md](https://github.com/BiomedicalDataMining/threshscore/blob/main/CONTRIBUTING.md).

## License

MIT: see [LICENSE](https://github.com/BiomedicalDataMining/threshscore/blob/main/LICENSE).
