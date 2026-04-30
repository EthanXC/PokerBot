"""Compare classifiers trained on synthetic features vs real game traces.

Two BluffClassifiers and two TiltClassifiers:
  A. Trained on hand-coded synthetic data (the version from the demo).
  B. Trained on real Leduc game traces (auto-labeled at showdown).

Both are evaluated on a held-out set of REAL traces. The expectation:
the real-trained model should generalize better to actual gameplay
because its training data has the actual feature distributions of play.

This is the test that distinguishes 'hand-tuned ML' from 'ML grounded
in reality.'
"""
from __future__ import annotations

import time

import numpy as np

from pokerbot.learning import BluffClassifier, TiltClassifier
from pokerbot.runtime import build_bluff_dataset, build_tilt_dataset
from tests.test_bluff_classifier import synth_bluff_data, auc as auc_fn
from tests.test_tilt_classifier import synth_tilt_data


def report(name: str, y_true, proba) -> None:
    acc = ((proba > 0.5) == (y_true > 0.5)).mean()
    au = auc_fn(np.asarray(y_true), np.asarray(proba))
    n = len(y_true)
    print(f"  {name:<40} acc={acc:.3f}  AUC={au:.3f}  (n={n})")


def main() -> None:
    print("=== Building real datasets from Leduc simulation ===")
    t0 = time.time()
    Xb_real, yb_real = build_bluff_dataset(n_hands_per_pairing=500, seed=0)
    print(f"  bluff: {len(Xb_real)} examples, {(yb_real == 1).mean():.1%} positive  "
          f"({time.time() - t0:.1f}s)")

    t0 = time.time()
    Xt_real, yt_real = build_tilt_dataset(n_hands_per_pairing=500, seed=0)
    print(f"  tilt:  {len(Xt_real)} examples, {(yt_real == 1).mean():.1%} positive  "
          f"({time.time() - t0:.1f}s)")

    # Train/test split
    rng = np.random.default_rng(0)

    def split(X, y, frac=0.2):
        idx = rng.permutation(len(X))
        n_test = int(len(X) * frac)
        return X[idx[n_test:]], y[idx[n_test:]], X[idx[:n_test]], y[idx[:n_test]]

    Xb_tr, yb_tr, Xb_te, yb_te = split(Xb_real, yb_real)
    Xt_tr, yt_tr, Xt_te, yt_te = split(Xt_real, yt_real)

    # Synthetic (the original training distribution)
    Xb_syn, yb_syn = synth_bluff_data(2000, seed=0)
    Xt_syn, yt_syn = synth_tilt_data(2000, seed=0)

    print("\n=== BLUFF CLASSIFIER ===")
    print("Evaluating on held-out REAL Leduc bluff/value examples:")

    clf_syn = BluffClassifier().fit(Xb_syn, yb_syn, epochs=400)
    p = clf_syn.predict_proba(Xb_te)
    report("Synthetic-trained, eval on real:", yb_te, p)

    clf_real = BluffClassifier().fit(Xb_tr, yb_tr, epochs=400)
    p = clf_real.predict_proba(Xb_te)
    report("Real-trained,      eval on real:", yb_te, p)

    print("\nFeature importances (real-trained):")
    for k, v in sorted(clf_real.feature_importances().items(), key=lambda kv: -kv[1]):
        print(f"  {k:<25} {v:.3f}")

    print("\n=== TILT CLASSIFIER ===")
    print("Evaluating on held-out REAL Leduc tilt/calm examples:")

    clf_syn = TiltClassifier().fit(Xt_syn, yt_syn, epochs=400)
    p = clf_syn.predict_proba(Xt_te)
    report("Synthetic-trained, eval on real:", yt_te, p)

    clf_real = TiltClassifier().fit(Xt_tr, yt_tr, epochs=400)
    p = clf_real.predict_proba(Xt_te)
    report("Real-trained,      eval on real:", yt_te, p)

    print("\nFeature importances (real-trained):")
    for k, v in sorted(clf_real.feature_importances().items(), key=lambda kv: -kv[1]):
        print(f"  {k:<25} {v:.3f}")


if __name__ == "__main__":
    main()
