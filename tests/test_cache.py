"""Tests for the disk cache module."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from pokerbot.cache import (
    save_strategy, load_strategy,
    save_classifier, load_classifier,
)
from pokerbot.learning import BluffClassifier
from tests.test_bluff_classifier import synth_bluff_data


class CacheRoundtripTest(unittest.TestCase):
    def test_strategy_roundtrip(self):
        s = {"K:": {"b": 0.7, "p": 0.3}, "Q:": {"b": 0.0, "p": 1.0}}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "strategy.pkl"
            save_strategy(s, path)
            loaded = load_strategy(path)
        self.assertEqual(loaded, s)

    def test_classifier_roundtrip(self):
        X, y = synth_bluff_data(200, seed=0)
        clf = BluffClassifier().fit(X, y, epochs=80)
        # Predict on something to lock in normalization params.
        before = clf.predict_proba(X[:10])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "clf.pkl"
            save_classifier(clf, path)
            loaded = load_classifier(path)

        after = loaded.predict_proba(X[:10])
        # Outputs must match exactly after roundtrip.
        for a, b in zip(before, after):
            self.assertAlmostEqual(float(a), float(b), places=12)

    def test_load_wrong_kind_errors(self):
        s = {"x": 1}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "s.pkl"
            save_strategy(s, path)
            with self.assertRaises(ValueError):
                load_classifier(path)


if __name__ == "__main__":
    unittest.main()
