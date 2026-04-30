"""Small MLP classifier — drop-in replacement for logistic regression.

Course topics: MLE & Optimization (gradient descent on cross-entropy),
non-linear function approximation.

Architecture: 1 hidden layer with tanh activation.
    forward:  x  ->  Linear  ->  tanh  ->  Linear  ->  sigmoid  ->  p
    loss:     -y log p - (1-y) log(1-p) + l2 * ||W||^2

Backprop is hand-written so the gradient flow is visible — no autograd
library used. Standardizes inputs at fit time (same as logistic).

Same fit / predict_proba interface as BluffClassifier, so it works in
cross_validate() and the calibration helpers without any wrapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MLPClassifier:
    hidden_dim: int = 16
    W1: np.ndarray = None
    b1: np.ndarray = None
    W2: np.ndarray = None
    b2: float = 0.0
    feature_means: np.ndarray = None
    feature_stds: np.ndarray = None
    history: list = field(default_factory=list)

    def fit(
        self,
        X,
        y,
        lr: float = 0.05,
        epochs: int = 800,
        l2: float = 0.001,
        seed: int = 0,
        verbose: bool = False,
    ) -> "MLPClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        N, D = X.shape

        # Standardize.
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        Xs = (X - self.feature_means) / self.feature_stds

        rng = np.random.default_rng(seed)
        # Xavier-ish init
        self.W1 = rng.normal(0, 1.0 / np.sqrt(D), size=(D, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(self.hidden_dim), size=self.hidden_dim)
        self.b2 = 0.0
        self.history = []

        for epoch in range(epochs):
            # ---- Forward ----
            z1 = Xs @ self.W1 + self.b1                  # (N, H)
            h = np.tanh(z1)                               # (N, H)
            z2 = h @ self.W2 + self.b2                    # (N,)
            p = 1.0 / (1.0 + np.exp(-np.clip(z2, -30, 30)))

            # ---- Loss ----
            eps = 1e-12
            ll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
            ll += 0.5 * l2 * (
                (self.W1 ** 2).sum() + (self.W2 ** 2).sum()
            )
            self.history.append(float(ll))

            # ---- Backward ----
            err = (p - y)                                 # (N,)  d_loss/d_z2
            grad_W2 = h.T @ err / N + l2 * self.W2        # (H,)
            grad_b2 = err.mean()

            d_h = np.outer(err, self.W2)                  # (N, H)
            d_z1 = d_h * (1 - h * h)                       # tanh derivative
            grad_W1 = Xs.T @ d_z1 / N + l2 * self.W1       # (D, H)
            grad_b1 = d_z1.mean(axis=0)

            # ---- Update ----
            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1

            if verbose and epoch % max(epochs // 10, 1) == 0:
                acc = ((p > 0.5) == (y > 0.5)).mean()
                print(f"epoch {epoch}: loss={ll:.4f} acc={acc:.3f}")

        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.W1 is None:
            raise RuntimeError("Call fit() first")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        Xs = (X - self.feature_means) / self.feature_stds
        h = np.tanh(Xs @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)
