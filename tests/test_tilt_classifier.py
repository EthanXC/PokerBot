"""Tests for TiltClassifier.

Generative synthetic data:
  - On-tilt:    big recent loss, VPIP jump, aggression jump, recent loss streak.
  - Not tilted: stable behavior, neutral or small losses.
"""
from __future__ import annotations

import unittest

import numpy as np

from pokerbot.learning.tilt_classifier import TiltClassifier, TILT_FEATURES


def synth_tilt_data(n: int, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = np.zeros((n, len(TILT_FEATURES)))
    for i, label in enumerate(y):
        if label == 1:
            # On tilt
            X[i, 0] = rng.normal(0.7, 0.2)        # recent_loss (high)
            X[i, 1] = rng.normal(0.15, 0.06)      # vpip_jump (looser)
            X[i, 2] = rng.normal(0.10, 0.05)      # aggression_jump (more aggro)
            X[i, 3] = rng.normal(2.0, 1.5)        # hands_since_loss (small)
            X[i, 4] = rng.normal(0.05, 0.03)      # three_bet_jump (loose 3-bet)
            X[i, 5] = rng.normal(4.0, 1.5)        # loss_streak
            X[i, 6] = rng.normal(0.05, 0.03)      # vol_increase
        else:
            X[i, 0] = rng.normal(0.1, 0.1)
            X[i, 1] = rng.normal(-0.01, 0.05)
            X[i, 2] = rng.normal(0.0, 0.04)
            X[i, 3] = rng.normal(15.0, 5.0)       # hands since loss high (=stable)
            X[i, 4] = rng.normal(0.0, 0.02)
            X[i, 5] = rng.normal(1.0, 1.0)
            X[i, 6] = rng.normal(0.0, 0.02)
    return X, y


class TiltClassifierTest(unittest.TestCase):
    def test_classifier_above_chance(self):
        X_train, y_train = synth_tilt_data(2000, seed=0)
        X_test, y_test = synth_tilt_data(500, seed=99)
        clf = TiltClassifier().fit(X_train, y_train, epochs=400)
        p = clf.predict_proba(X_test)
        acc = ((p > 0.5) == (y_test > 0.5)).mean()
        self.assertGreater(acc, 0.85, f"tilt accuracy too low: {acc}")

    def test_loss_history_decreasing(self):
        X, y = synth_tilt_data(1000, seed=0)
        clf = TiltClassifier().fit(X, y, epochs=200)
        self.assertLess(clf.history[-1], clf.history[0])

    def test_feature_importance_recent_loss_matters(self):
        X, y = synth_tilt_data(2000, seed=0)
        clf = TiltClassifier().fit(X, y, epochs=300)
        imps = clf.feature_importances()
        # recent_loss and loss_streak should outweigh hands_since_loss alone.
        # All of them matter; we just check the most signal-rich ones aren't
        # tiny.
        self.assertGreater(imps["recent_loss"], 0.05)
        self.assertGreater(imps["vpip_jump"], 0.05)


if __name__ == "__main__":
    unittest.main()
