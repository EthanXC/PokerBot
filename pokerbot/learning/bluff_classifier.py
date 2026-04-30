"""Bluff classifier — learned P(opponent is bluffing | context).

This is the core "AI catches humans doing human things" component.

Course topics covered: Probabilistic Reasoning (Bayesian classification),
MLE & Optimization (logistic regression fit by gradient descent on the
log-likelihood), Learning.

WHAT IT DOES
------------
At every betting decision where the bot is facing a bet/raise from an
opponent, we want a probability estimate:

    P(bluff | context features)

Features we use (stack-agnostic, work in any pot):
    f_bet_size_ratio       — bet_size / pot, on a log scale
    f_board_strength       — how connected/coordinated the board is (0..1)
    f_prior_aggression     — opponent's recent aggression freq (0..1)
    f_position             — 1.0 if opponent has position, 0.0 otherwise
    f_pot_committed        — chips opp has put in / their starting stack
    f_n_bets_this_street   — count of bets/raises this street
    f_overbet              — 1.0 if bet > pot, else 0.0
    f_donk_lead            — 1.0 if opp donk-led (bet out of position into preflop raiser)

These are real, measurable from a hand history, and don't require seeing
the opponent's cards. The label (bluff vs value) we get from showdowns or
synthetic training data.

MODEL
-----
Logistic regression with L2 regularization, fit by full-batch gradient
descent on the negative log-likelihood. Pure NumPy — no sklearn — so the
training loop is visible.

    P(bluff | x) = sigmoid(w . x + b)

    Loss = -1/N * sum_n [ y_n * log p_n + (1 - y_n) * log (1 - p_n) ]
           + lambda/2 * ||w||^2

NOTE
----
We probabilistically combine this output with the bet-size and pot-odds
math when deciding whether to call:

    Call if  P(bluff) * P(beat-non-bluff-range) + (1 - P(bluff)) * P(beat-value-range)
             > pot_odds

(In the bot, the BR layer handles this; the bluff prob feeds the "perceived
opponent strategy" we hand to BR.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Feature names in canonical order.
# Indices 0-7 are the original "in-the-moment" features.
# Indices 8-11 are the temporal/context features added in the v2 upgrade —
# they look at the BETTOR's history to provide longer-horizon signal.
FEATURES = [
    # In-the-moment features.
    "bet_size_ratio",
    "board_strength",
    "prior_aggression",
    "position",
    "pot_committed",
    "n_bets_this_street",
    "overbet",
    "donk_lead",
    # Temporal / opponent-history features.
    "bettor_recent_bluff_rate",
    "bettor_showdown_rate",
    "hand_aggression_so_far",
    "street_aggression_pace",
]


@dataclass
class BluffClassifier:
    """Logistic regression for P(bluff | features).

    Train via fit(); apply via predict_proba().

    Attributes:
        weights:  numpy array of shape (n_features,)
        bias:     scalar
        feature_means / feature_stds: standardization params learned at fit time
        history:  list of training losses
    """

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
        epochs: int = 500,
        l2: float = 0.001,
        verbose: bool = False,
    ) -> "BluffClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of rows")

        # Standardize features (zero mean, unit variance) so gradient descent
        # converges cleanly even when features have different scales.
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-8
        Xs = (X - self.feature_means) / self.feature_stds

        N, D = Xs.shape
        self.weights = np.zeros(D)
        self.bias = 0.0
        self.history = []

        for epoch in range(epochs):
            # Forward
            z = Xs @ self.weights + self.bias
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

            # Loss (cross-entropy + L2)
            eps = 1e-12
            ll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
            ll += 0.5 * l2 * (self.weights @ self.weights)
            self.history.append(float(ll))

            # Gradients
            err = p - y                                    # (N,)
            grad_w = Xs.T @ err / N + l2 * self.weights    # (D,)
            grad_b = err.mean()

            # GD step
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

            if verbose and epoch % max(epochs // 10, 1) == 0:
                acc = ((p > 0.5) == (y > 0.5)).mean()
                print(f"epoch {epoch}: loss={ll:.4f} acc={acc:.3f}")

        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit() before predict_proba()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        Xs = (X - self.feature_means) / self.feature_stds
        z = Xs @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def partial_fit(self, X, y, lr: float = 0.02, l2: float = 0.001) -> None:
        """One online gradient step on a small batch.

        Uses the *standardization* learned at fit() time. If you've never
        called fit(), this errors. Useful for: as the bot plays, after every
        showdown we observe a new (features, was_bluff) pair, and we
        nudge the classifier to better fit it.
        """
        if self.weights is None:
            raise RuntimeError("Call fit() before partial_fit() — need standardization")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X[None, :]
        Xs = (X - self.feature_means) / self.feature_stds
        z = Xs @ self.weights + self.bias
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = p - y
        N = len(y)
        grad_w = Xs.T @ err / N + l2 * self.weights
        grad_b = err.mean()
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b

    def feature_importances(self) -> dict:
        """Return |weight| / std-scale for each named feature.

        Larger = more influence on the prediction. Useful as a sanity check
        — features like 'overbet' and 'bet_size_ratio' should dominate;
        features like 'position' alone should be weaker.
        """
        if self.weights is None:
            raise RuntimeError("Not fit yet")
        return {name: float(abs(self.weights[i])) for i, name in enumerate(FEATURES)}
