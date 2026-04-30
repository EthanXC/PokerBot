"""Tests for the validation module: AUC, Brier, ECE, reliability, CV."""
from __future__ import annotations

import unittest

import numpy as np

from pokerbot.learning import (
    BluffClassifier,
    auc,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
    cross_validate,
)
from tests.test_bluff_classifier import synth_bluff_data


class AUCTest(unittest.TestCase):
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(auc(y, p), 1.0)

    def test_random_predictions(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=2000)
        p = rng.random(size=2000)
        self.assertAlmostEqual(auc(y, p), 0.5, delta=0.05)

    def test_inverted_predictions(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(auc(y, p), 0.0)


class BrierTest(unittest.TestCase):
    def test_perfect_zero_brier(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.0, 1.0, 0.0, 1.0])
        self.assertEqual(brier_score(y, p), 0.0)

    def test_max_brier(self):
        y = np.array([0, 1])
        p = np.array([1.0, 0.0])
        self.assertEqual(brier_score(y, p), 1.0)


class ECETest(unittest.TestCase):
    def test_perfectly_calibrated_zero_ece(self):
        # 1000 samples, predictions match empirical frequencies
        rng = np.random.default_rng(0)
        n = 5000
        p = rng.random(size=n)
        y = (rng.random(size=n) < p).astype(int)
        ece = expected_calibration_error(y, p, n_bins=10)
        # Sample variance only — should be tiny
        self.assertLess(ece, 0.05)

    def test_overconfident_high_ece(self):
        # All predictions say 0.9, but only 50% are positive
        n = 1000
        y = np.array([0, 1] * (n // 2))
        p = np.full(n, 0.9)
        ece = expected_calibration_error(y, p)
        self.assertAlmostEqual(ece, 0.4, delta=0.05)


class ReliabilityDiagramTest(unittest.TestCase):
    def test_returns_curve(self):
        rng = np.random.default_rng(0)
        n = 500
        p = rng.random(size=n)
        y = (rng.random(size=n) < p).astype(int)
        curve = reliability_diagram(y, p, n_bins=10)
        self.assertEqual(len(curve.bin_centers), 10)
        self.assertEqual(len(curve.bin_counts), 10)
        # Total examples in bins == n
        self.assertEqual(int(curve.bin_counts.sum()), n)


class CrossValidateTest(unittest.TestCase):
    def test_kfold_runs_and_reports(self):
        X, y = synth_bluff_data(800, seed=0)
        report = cross_validate(
            model_factory=BluffClassifier,
            X=X, y=y,
            k=5, seed=0,
            fit_kwargs={"epochs": 100},
        )
        self.assertEqual(len(report.aucs), 5)
        # AUC should be well above chance on synthetic separable data
        self.assertGreater(report.auc_mean, 0.8)
        # Std should be small (synthetic data is consistent)
        self.assertLess(report.auc_std, 0.05)


if __name__ == "__main__":
    unittest.main()
