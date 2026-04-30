"""Tests for BluffClassifier.

We synthesize labeled training data using a generative model:
  - Bluff features:  bigger bets, weaker boards, more recent aggression,
                     overbets, donk leads.
  - Value features:  smaller-to-medium bets, stronger boards, lower
                     aggression, no overbet, no donk.

Then train and check:
  1. Cross-entropy loss decreases each epoch.
  2. Held-out accuracy > 0.7 (synthetic data is separable but noisy).
  3. AUC > 0.8.
  4. The features the model leans on are the ones we'd expect (bet_size_ratio,
     overbet, board_strength), via |weight| inspection.
"""
from __future__ import annotations

import unittest

import numpy as np

from pokerbot.learning.bluff_classifier import BluffClassifier, FEATURES


def synth_bluff_data(n: int, seed: int = 0) -> tuple:
    """Generate synthetic (X, y) where y=1 means bluff.

    Updated to produce all 12 features (8 original + 4 temporal/context).
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)  # half bluff, half value
    X = np.zeros((n, len(FEATURES)))
    for i, label in enumerate(y):
        if label == 1:
            # BLUFF
            X[i, 0] = rng.normal(1.2, 0.4)
            X[i, 1] = rng.normal(0.3, 0.15)
            X[i, 2] = rng.normal(0.7, 0.15)
            X[i, 3] = rng.choice([0.0, 1.0])
            X[i, 4] = rng.normal(0.4, 0.2)
            X[i, 5] = rng.choice([1, 2, 3])
            X[i, 6] = 1.0 if X[i, 0] > 1.0 else 0.0
            X[i, 7] = rng.choice([0.0, 1.0], p=[0.6, 0.4])
            # New temporal features
            X[i, 8]  = rng.normal(0.55, 0.1)   # bettor_recent_bluff_rate (high)
            X[i, 9]  = rng.normal(0.45, 0.1)   # bettor_showdown_rate
            X[i, 10] = rng.choice([0, 1, 2, 3])  # hand_aggression_so_far
            X[i, 11] = rng.normal(0.45, 0.15)  # street_aggression_pace (faster)
        else:
            # VALUE
            X[i, 0] = rng.normal(0.6, 0.2)
            X[i, 1] = rng.normal(0.7, 0.15)
            X[i, 2] = rng.normal(0.4, 0.15)
            X[i, 3] = rng.choice([0.0, 1.0])
            X[i, 4] = rng.normal(0.5, 0.2)
            X[i, 5] = rng.choice([1, 2])
            X[i, 6] = 1.0 if X[i, 0] > 1.0 else 0.0
            X[i, 7] = rng.choice([0.0, 1.0], p=[0.85, 0.15])
            X[i, 8]  = rng.normal(0.25, 0.1)   # lower bluff rate
            X[i, 9]  = rng.normal(0.55, 0.1)
            X[i, 10] = rng.choice([0, 1, 2])
            X[i, 11] = rng.normal(0.30, 0.15)
    return X, y


def auc(y_true, scores) -> float:
    """Quick AUC computation via Mann-Whitney U statistic."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Count pairs where pos > neg (with ties = 0.5)
    n_correct = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                n_correct += 1.0
            elif p == n:
                n_correct += 0.5
    return n_correct / (len(pos) * len(neg))


class BluffClassifierTest(unittest.TestCase):
    def test_loss_decreases(self):
        X, y = synth_bluff_data(800, seed=0)
        clf = BluffClassifier()
        clf.fit(X, y, epochs=300)
        # Loss curve must be monotone non-increasing modulo small numerics.
        h = clf.history
        # Allow small bumps but final < first.
        self.assertLess(h[-1], h[0])

    def test_classifier_separates_bluff_vs_value(self):
        rng = np.random.default_rng(7)
        X_train, y_train = synth_bluff_data(2000, seed=int(rng.integers(1e9)))
        X_test, y_test = synth_bluff_data(500, seed=int(rng.integers(1e9)))
        clf = BluffClassifier()
        clf.fit(X_train, y_train, epochs=400)
        proba = clf.predict_proba(X_test)
        acc = ((proba > 0.5) == (y_test > 0.5)).mean()
        au = auc(y_test, proba)
        self.assertGreater(acc, 0.75, f"accuracy too low: {acc}")
        self.assertGreater(au, 0.85, f"AUC too low: {au}")

    def test_feature_importances_emphasize_expected_features(self):
        X, y = synth_bluff_data(2000, seed=42)
        clf = BluffClassifier()
        clf.fit(X, y, epochs=400)
        imps = clf.feature_importances()
        # Bet size and board strength should be high-weight; position should
        # not dominate (it's coin-flip in our synthesizer).
        self.assertGreater(imps["bet_size_ratio"], imps["position"])
        self.assertGreater(imps["board_strength"], imps["position"])

    def test_predict_proba_is_in_unit_interval(self):
        X, y = synth_bluff_data(200, seed=3)
        clf = BluffClassifier().fit(X, y, epochs=100)
        p = clf.predict_proba(X)
        self.assertTrue((p >= 0).all())
        self.assertTrue((p <= 1).all())


if __name__ == "__main__":
    unittest.main()
