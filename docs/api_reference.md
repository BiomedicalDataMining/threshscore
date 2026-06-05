# API Reference

## Top-level functions

These are importable directly from `threshscore`.

---

### `threshscore.score_from_metrics`

```python
threshscore.score_from_metrics(
    sens: float,
    spec: float,
    *,
    s_thr: float = 0.8,
    p_thr: float = 0.6,
    gate: Union[str, BaseGate] = "arctan",
    gate_params: Optional[dict] = None,
) -> float
```

Compute the centering-based score directly from sensitivity and specificity values.

Returns **exactly 0.5** when `sens == s_thr` and `spec == p_thr`, > 0.5 when both
metrics exceed their thresholds, and < 0.5 when either falls short.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sens` | float | (none) | Achieved sensitivity in [0, 1] |
| `spec` | float | (none) | Achieved specificity in [0, 1] |
| `s_thr` | float | `0.8` | Minimum required sensitivity |
| `p_thr` | float | `0.6` | Minimum required specificity |
| `gate` | str or BaseGate | `"arctan"` | Gate function |
| `gate_params` | dict | `None` | Override gate defaults |

**Returns** `float`: score in [0, 1]

---

### `threshscore.score_from_predictions`

```python
threshscore.score_from_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    s_thr: float = 0.8,
    p_thr: float = 0.6,
    gate: Union[str, BaseGate] = "arctan",
    gate_params: Optional[dict] = None,
) -> ScoreResult
```

Compute the centering-based score from binary predictions.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | array-like | (none) | Ground-truth binary labels (0 or 1) |
| `y_pred` | array-like | (none) | Predicted binary labels (0 or 1) |
| `s_thr` | float | `0.8` | Minimum required sensitivity |
| `p_thr` | float | `0.6` | Minimum required specificity |
| `gate` | str or BaseGate | `"arctan"` | Gate function |
| `gate_params` | dict | `None` | Override gate defaults |

**Returns** `ScoreResult`

---

### `threshscore.score_from_scores`

```python
threshscore.score_from_scores(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float = 0.5,
    *,
    s_thr: float = 0.8,
    p_thr: float = 0.6,
    gate: Union[str, BaseGate] = "arctan",
    gate_params: Optional[dict] = None,
) -> ScoreResult
```

Compute the centering-based score from probability scores.  Binarises `y_score` at
`threshold` to get predictions.  The full metrics object (including ROC AUC and PR AUC)
is available on the result.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `y_true` | array-like | (none) | Ground-truth binary labels (0 or 1) |
| `y_score` | array-like | (none) | Predicted probabilities in [0, 1] |
| `threshold` | float | `0.5` | Decision threshold for binarisation |
| `s_thr` | float | `0.8` | Minimum required sensitivity |
| `p_thr` | float | `0.6` | Minimum required specificity |
| `gate` | str or BaseGate | `"arctan"` | Gate function |
| `gate_params` | dict | `None` | Override gate defaults |

**Returns** `ScoreResult`

---

### `threshscore.compute_metrics`

```python
threshscore.compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: Optional[ArrayLike] = None,
) -> ClassificationMetrics
```

Compute all standard binary classification metrics.

**Returns** `ClassificationMetrics`

---

### `threshscore.sweep`

```python
threshscore.sweep(
    y_true: ArrayLike,
    y_score: ArrayLike,
    n_thresholds: int = 101,
) -> ThresholdSweep
```

Sweep classification thresholds from 0.0 to 1.0 and compute metrics at each.

**Returns** `ThresholdSweep`

---

## Data Classes

### `ScoreResult`

Frozen dataclass returned by `score_from_predictions()` and `score_from_scores()`.

| Field | Type | Description |
|-------|------|-------------|
| `score` | float | Centering-based score in [0, 1]; 0.5 at operating point |
| `sensitivity` | float | Achieved sensitivity |
| `specificity` | float | Achieved specificity |
| `sensitivity_gate` | float | Gate value g_s = gate(sens, s_thr) |
| `specificity_gate` | float | Gate value g_p = gate(spec, p_thr) |
| `sensitivity_threshold` | float | Configured sensitivity threshold (s_thr) |
| `specificity_threshold` | float | Configured specificity threshold (p_thr) |
| `gate_name` | str | Name of the gate function used |
| `metrics` | ClassificationMetrics | Full metrics object |

---

### `ClassificationMetrics`

Frozen dataclass returned by `compute_metrics()`.

| Field | Type | Description |
|-------|------|-------------|
| `sensitivity` | float | True positive rate (= recall) |
| `specificity` | float | True negative rate |
| `precision` | float | Positive predictive value |
| `recall` | float | Same as sensitivity |
| `f1` | float | Harmonic mean of precision and recall |
| `accuracy` | float | (TP + TN) / N |
| `roc_auc` | float or None | Area under ROC curve; `None` if no `y_score` |
| `pr_auc` | float or None | Area under PR curve; `None` if no `y_score` |
| `confusion_matrix` | ndarray | 2×2 array `[[TN, FP], [FN, TP]]` |
| `n_samples` | int | Total sample count |
| `n_positive` | int | Positive sample count |
| `n_negative` | int | Negative sample count |

---

### `ThresholdSweep`

Frozen dataclass returned by `threshscore.sweep()`.

| Field | Type | Description |
|-------|------|-------------|
| `points` | list[ThresholdPoint] | Per-threshold metric objects |
| `thresholds` | ndarray | Threshold values evaluated |
| `sensitivities` | ndarray | Sensitivity at each threshold |
| `specificities` | ndarray | Specificity at each threshold |

**Methods**

```python
sweep.best_by_f1() -> ThresholdPoint
sweep.best_by_sensitivity_at_specificity(min_specificity: float) -> Optional[ThresholdPoint]
sweep.best_by_specificity_at_sensitivity(min_sensitivity: float) -> Optional[ThresholdPoint]
```

---

### `ThresholdPoint`

Frozen dataclass: metrics at a single operating threshold.

| Field | Type |
|-------|------|
| `threshold` | float |
| `sensitivity` | float |
| `specificity` | float |
| `precision` | float |
| `f1` | float |
| `accuracy` | float |
| `n_predicted_positive` | int |

---

## Scoring Formula

### Arctan (default)

```
q     = 0.5 * (sens + spec)
g_s   = 0.5 + (1/π) * arctan(k * (sens − s_thr))
g_p   = 0.5 + (1/π) * arctan(k * (spec − p_thr))
gate  = g_s * g_p
score = clip(0.5 + 0.5*(gate − 0.25) + 0.5*(q − 0.5*(s_thr + p_thr)), 0, 1)
```

**Key property:** score = 0.5 exactly when sens = s_thr AND spec = p_thr (since g_s = g_p = 0.5 → gate = 0.25 and q = q_thr).

### Sigmoid

Same centering formula with `g_s = sigmoid(k * (sens − s_thr))`.  `sigmoid(0) = 0.5`, so `gate_thr = 0.25`: identical structure to arctan.

### relu_clip and linear_clip

Same centering formula as arctan: `gate(thr, thr) = 0.5` for both, so `gate_thr = 0.25`.

---

## Plotting (`threshscore.plot`)

All plot functions return `(matplotlib.figure.Figure, matplotlib.axes.Axes)` and accept an optional `ax` keyword argument.

```python
import threshscore

# ROC curve
fig, ax = threshscore.plot.plot_roc(y_true, y_score, *, ax=None, label=None)

# Precision-Recall curve
fig, ax = threshscore.plot.plot_pr(y_true, y_score, *, ax=None, label=None)

# Threshold vs. sensitivity / specificity / F1
fig, ax = threshscore.plot.plot_threshold(sweep_or_none, *, y_true=None, y_score=None, ax=None)

# Confusion matrix heatmap
fig, ax = threshscore.plot.plot_confusion(cm, *, normalise=False, labels=None, ax=None)

# Score heatmap over sensitivity × specificity (centering formula visualised)
fig, ax = threshscore.plot.plot_heatmap(
    *,
    sensitivity_threshold=0.8,
    specificity_threshold=0.6,
    gate="arctan",
    target_sensitivity=None,
    target_specificity=None,
    n_grid=200,
    show_contours=True,
    contour_levels=None,
    ax=None,
)

# Gate curve
fig, ax = threshscore.plot.plot_gate(gate="arctan", *, threshold=0.5, gate_params=None, ax=None)
```

---

## Gate Functions (`threshscore.gates`)

### Built-in gate functions

| Name | Class | Key params | `gate_thr` | Notes |
|------|-------|-----------|------------|-------|
| `"arctan"` | `ArctanGate` | `k=25` | 0.25 | Default; smooth S-curve |
| `"sigmoid"` | `SigmoidGate` | `k=15` | 0.25 | Classic logistic |
| `"relu_clip"` | `ReluClipGate` | (none) | 0.25 | Hard cliff below threshold |
| `"linear_clip"` | `LinearClipGate` | (none) | 0.25 | Partial credit below threshold |

### Interface

Each `BaseGate` subclass must implement:

```python
def __call__(self, value: float, threshold: float, **params) -> float:
    """Single-metric gate: maps (value, threshold) → [0, 1]. Must return 0.5 at threshold."""
```

### Registry functions

```python
threshscore.gates.list_gates() -> list[str]
threshscore.gates.get(name: str) -> BaseGate
threshscore.gates.register(name: str, gate: BaseGate) -> None
threshscore.gates.register_class(name: str, cls: type[BaseGate]) -> None
```
