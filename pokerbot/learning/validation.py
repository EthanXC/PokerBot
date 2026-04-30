"""Cross-validation utilities + reliability/calibration analysis.

Course topics: Probabilistic Reasoning, Optimization & Validation.

cross_validate
--------------
Generic K-fold for any classifier with .fit(X, y, **kwargs) and
.predict_proba(X). Returns per-fold AUC + accuracy + Brier score so
we can see variance across folds. Single 80/20 splits hide variance.

reliability_diagram
-------------------
Bin predicted probabilities and compute the EMPIRICAL fraction of
positives in each bin. A perfectly calibrated model has empirical
fraction equal to the bin's mean predicted probability. The
reliability gap (or expected calibration error, ECE) is the
average |bin_pred - bin_actual| weighted by bin size.

Brier score
-----------
Mean squared error of predicted probability vs binary label.
A proper scoring rule: minimized when predictions are calibrated AND
discriminative. Lower = better.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def auc(y_true, scores) -> float:
    """ROC-AUC by Mann-Whitney U. Vectorized version of the test helper."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # For each pos, count negs it ranks above (with ties = 0.5)
    diffs = pos[:, None] - neg[None, :]
    return float((diffs > 0).sum() + 0.5 * (diffs == 0).sum()) / (len(pos) * len(neg))


def brier_score(y_true, proba) -> float:
    """Mean squared error of predicted probability vs binary label."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    return float(((proba - y_true) ** 2).mean())


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Average |empirical_freq - mean_predicted_prob| weighted by bin size."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    if len(y_true) == 0:
        return 0.0
    # Bin edges. We use closed-on-the-right; the rightmost bin includes 1.0.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (proba >= lo) & (proba <= hi)
        else:
            mask = (proba >= lo) & (proba < hi)
        bin_n = mask.sum()
        if bin_n == 0:
            continue
        bin_pred = proba[mask].mean()
        bin_actual = y_true[mask].mean()
        ece += (bin_n / n) * abs(bin_pred - bin_actual)
    return float(ece)


@dataclass
class ReliabilityCurve:
    bin_centers: np.ndarray   # mean predicted prob in each bin
    bin_actuals: np.ndarray   # empirical fraction of positives in each bin
    bin_counts: np.ndarray    # # examples in each bin
    ece: float
    brier: float

    def show_text(self) -> str:
        lines = [
            f"  ECE = {self.ece:.4f}    Brier = {self.brier:.4f}",
            "  bin   pred    actual    n",
        ]
        for c, a, n in zip(self.bin_centers, self.bin_actuals, self.bin_counts):
            if n == 0:
                lines.append(f"  ----  ----   ----     0")
                continue
            lines.append(f"  {c:.2f}  {c:.3f}  {a:.3f}  {int(n):>5d}")
        return "\n".join(lines)


def reliability_diagram(y_true, proba, n_bins: int = 10) -> ReliabilityCurve:
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    actuals = []
    counts = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (proba >= lo) & (proba <= hi)
        else:
            mask = (proba >= lo) & (proba < hi)
        bin_n = mask.sum()
        counts.append(bin_n)
        if bin_n == 0:
            centers.append((lo + hi) / 2)
            actuals.append(np.nan)
        else:
            centers.append(float(proba[mask].mean()))
            actuals.append(float(y_true[mask].mean()))
    return ReliabilityCurve(
        bin_centers=np.array(centers),
        bin_actuals=np.array(actuals),
        bin_counts=np.array(counts),
        ece=expected_calibration_error(y_true, proba, n_bins),
        brier=brier_score(y_true, proba),
    )


@dataclass
class CrossValReport:
    aucs: list
    accuracies: list
    briers: list
    eces: list

    @property
    def auc_mean(self) -> float:
        return float(np.mean(self.aucs))

    @property
    def auc_std(self) -> float:
        return float(np.std(self.aucs))

    def show_text(self) -> str:
        return (
            f"  AUC      = {self.auc_mean:.3f} +- {self.auc_std:.3f}  "
            f"(folds={['%.3f' % a for a in self.aucs]})\n"
            f"  Accuracy = {np.mean(self.accuracies):.3f} +- {np.std(self.accuracies):.3f}\n"
            f"  Brier    = {np.mean(self.briers):.4f} +- {np.std(self.briers):.4f}\n"
            f"  ECE      = {np.mean(self.eces):.4f} +- {np.std(self.eces):.4f}"
        )


def cross_validate(
    model_factory,
    X,
    y,
    k: int = 5,
    seed: int = 0,
    fit_kwargs: dict | None = None,
) -> CrossValReport:
    """Run k-fold CV on a fresh classifier per fold.

    `model_factory` is a zero-arg callable returning a new untrained classifier;
    we rebuild per fold so weights aren't shared across folds.
    `fit_kwargs` are passed to clf.fit.
    """
    fit_kwargs = fit_kwargs or {}
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1

    aucs = []
    accs = []
    briers = []
    eces = []

    start = 0
    for f in range(k):
        end = start + fold_sizes[f]
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        start = end

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        clf = model_factory()
        clf.fit(X_tr, y_tr, **fit_kwargs)
        p = clf.predict_proba(X_te)

        aucs.append(auc(y_te, p))
        accs.append(((p > 0.5).astype(int) == y_te).mean())
        briers.append(brier_score(y_te, p))
        eces.append(expected_calibration_error(y_te, p))

    return CrossValReport(aucs=aucs, accuracies=accs, briers=briers, eces=eces)
