"""Tilt classifier — learned P(opponent is on tilt | recent behavioral signals).

Replaces the heuristic in OpponentStats.tilt_score() with a model that's
fit from labeled data (synthetic or harvested from hand histories where
we have ground truth for "did this player just dump chips after a bad beat").

Course topics: Probabilistic Reasoning, MLE & Optimization (logistic
regression by gradient descent).

FEATURES
--------
    f_recent_loss          — chips lost in last K hands, normalized
    f_vpip_jump            — recent_VPIP - lifetime_VPIP (positive = looser lately)
    f_aggression_jump      — recent_AGG - lifetime_AGG
    f_hands_since_loss     — small if a big loss was recent (more on-tilt)
    f_3bet_jump            — same idea for preflop 3-bet rate
    f_loss_streak          — number of consecutive losing hands
    f_vol_increase         — recent stddev of VPIP minus lifetime stddev

These features are derived from a windowed buffer of the opponent's recent
behavior (already tracked in OpponentStats). They're all real-time
computable from a hand log.

MODEL
-----
Same logistic regression as BluffClassifier — small, fast, interpretable,
fittable on tens of thousands of synthetic examples in under a second.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


TILT_FEATURES = [
    "recent_loss",
    "vpip_jump",
    "aggression_jump",
    "hands_since_loss",
    "three_bet_jump",
    "loss_streak",
    "vol_increase",
]


@dataclass
class TiltClassifier:
    weights: np.ndarray = field(default=None)
    bias: float = 0.0
    feature_means: np.ndarray = field(default=None)
    feature_stds: np.ndarray = field(default=None)
    history: list = field(default_factory=list)

    def fit(
        self,
        X,
        y,
        lr: float = 0.1,
        epochs: int = 800,
        l2: float = 0.001,
        verbose: bool = False,
    ) -> "TiltClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same length")

        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        Xs = (X - self.feature_means) / self.feature_stds

        N, D = Xs.shape
        self.weights = np.zeros(D)
        self.bias = 0.0
        self.history = []

        for epoch in range(epochs):
            z = Xs @ self.weights + self.bias
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

            eps = 1e-12
            ll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
            ll += 0.5 * l2 * (self.weights @ self.weights)
            self.history.append(float(ll))

            err = p - y
            grad_w = Xs.T @ err / N + l2 * self.weights
            grad_b = err.mean()

            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

            if verbose and epoch % max(epochs // 10, 1) == 0:
                acc = ((p > 0.5) == (y > 0.5)).mean()
                print(f"epoch {epoch}: loss={ll:.4f} acc={acc:.3f}")

        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit() first")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        Xs = (X - self.feature_means) / self.feature_stds
        z = Xs @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def feature_importances(self) -> dict:
        if self.weights is None:
            raise RuntimeError("Not fit yet")
        return {name: float(abs(self.weights[i])) for i, name in enumerate(TILT_FEATURES)}
