"""Regression test: real-trained classifiers must beat synthetic-trained
classifiers on real held-out Leduc data.

This is the test that asserts our 'ground the AI in reality' improvement
is actually paying off. If anyone changes the feature extractor or the
heuristic players in a way that breaks transfer, this will fail.
"""
from __future__ import annotations

import unittest

import numpy as np

from pokerbot.learning import BluffClassifier, TiltClassifier
from pokerbot.runtime import build_bluff_dataset, build_tilt_dataset
from tests.test_bluff_classifier import synth_bluff_data, auc as auc_fn
from tests.test_tilt_classifier import synth_tilt_data


def _train_test_split(X, y, frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * frac)
    return X[idx[n_test:]], y[idx[n_test:]], X[idx[:n_test]], y[idx[:n_test]]


class RealDataAdvantageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_real, cls.y_real = build_bluff_dataset(n_hands_per_pairing=300, seed=0)
        cls.Xb_tr, cls.yb_tr, cls.Xb_te, cls.yb_te = _train_test_split(
            cls.X_real, cls.y_real
        )

        cls.Xt_real, cls.yt_real = build_tilt_dataset(n_hands_per_pairing=300, seed=0)
        cls.Xt_tr, cls.yt_tr, cls.Xt_te, cls.yt_te = _train_test_split(
            cls.Xt_real, cls.yt_real
        )

    def test_bluff_real_beats_synthetic(self):
        Xb_syn, yb_syn = synth_bluff_data(1500, seed=0)

        clf_syn = BluffClassifier().fit(Xb_syn, yb_syn, epochs=300)
        clf_real = BluffClassifier().fit(self.Xb_tr, self.yb_tr, epochs=300)

        auc_syn = auc_fn(self.yb_te, clf_syn.predict_proba(self.Xb_te))
        auc_real = auc_fn(self.yb_te, clf_real.predict_proba(self.Xb_te))

        self.assertGreater(
            auc_real, auc_syn + 0.05,
            f"real-trained bluff classifier must beat synthetic-trained by >=0.05 AUC; "
            f"got real={auc_real:.3f}, synthetic={auc_syn:.3f}"
        )
        # And real-trained must be meaningfully above chance.
        self.assertGreater(auc_real, 0.6, f"real-trained AUC too low: {auc_real}")

    def test_tilt_real_beats_synthetic(self):
        Xt_syn, yt_syn = synth_tilt_data(1500, seed=0)

        clf_syn = TiltClassifier().fit(Xt_syn, yt_syn, epochs=300)
        clf_real = TiltClassifier().fit(self.Xt_tr, self.yt_tr, epochs=300)

        auc_syn = auc_fn(self.yt_te, clf_syn.predict_proba(self.Xt_te))
        auc_real = auc_fn(self.yt_te, clf_real.predict_proba(self.Xt_te))

        self.assertGreater(
            auc_real, auc_syn + 0.02,
            f"real-trained tilt classifier must beat synthetic-trained by >=0.02 AUC; "
            f"got real={auc_real:.3f}, synthetic={auc_syn:.3f}"
        )
        self.assertGreater(auc_real, 0.75, f"real-trained tilt AUC too low: {auc_real}")


if __name__ == "__main__":
    unittest.main()
