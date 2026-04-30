"""Phase-2 deliverable: shows the AUC/calibration improvements.

For both Bluff and Tilt classifiers:
  1. Train logistic regression with the OLD 8 features (baseline).
  2. Train logistic regression with the NEW 12 features.
  3. Train an MLP with the NEW 12 features.

For each, run 5-fold CV on real Leduc traces and report:
  - AUC (mean +/- std)
  - Brier score
  - Expected calibration error (ECE)
  - Reliability diagram

This makes the impact of the upgrade concrete.
"""
from __future__ import annotations

import time

import numpy as np

from pokerbot.learning import (
    BluffClassifier,
    TiltClassifier,
    cross_validate,
    reliability_diagram,
)
from pokerbot.learning.mlp_classifier import MLPClassifier
from pokerbot.runtime import build_bluff_dataset, build_tilt_dataset


def _truncate_to_8(X):
    """Strip the temporal columns to recreate the v1 input space."""
    return X[:, :8]


def main() -> None:
    print("Building real Leduc datasets...")
    t0 = time.time()
    Xb, yb = build_bluff_dataset(n_hands_per_pairing=400, seed=0)
    Xt, yt = build_tilt_dataset(n_hands_per_pairing=400, seed=0)
    print(f"  bluff: {len(Xb)} examples ({(yb == 1).mean():.1%} bluff)  "
          f"tilt: {len(Xt)} examples ({(yt == 1).mean():.1%} tilted)  "
          f"({time.time() - t0:.1f}s)")

    print("\n=== BLUFF CLASSIFIER (5-fold CV) ===")

    print("\n[1] Logistic regression, OLD 8 features:")
    rep = cross_validate(BluffClassifier, _truncate_to_8(Xb), yb, k=5,
                         fit_kwargs={"epochs": 400})
    print(rep.show_text())

    print("\n[2] Logistic regression, NEW 12 features:")
    rep_lr_v2 = cross_validate(BluffClassifier, Xb, yb, k=5,
                               fit_kwargs={"epochs": 400})
    print(rep_lr_v2.show_text())

    print("\n[3] MLP (16 hidden units), NEW 12 features:")
    rep_mlp = cross_validate(lambda: MLPClassifier(hidden_dim=16), Xb, yb, k=5,
                             fit_kwargs={"epochs": 800, "lr": 0.05})
    print(rep_mlp.show_text())

    print("\nReliability diagram (best model: MLP on full feature set, "
          "single hold-out fold):")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xb))
    n_te = len(Xb) // 5
    Xtr, ytr = Xb[idx[n_te:]], yb[idx[n_te:]]
    Xte, yte = Xb[idx[:n_te]], yb[idx[:n_te]]
    mlp = MLPClassifier(hidden_dim=16).fit(Xtr, ytr, epochs=800, lr=0.05)
    p = mlp.predict_proba(Xte)
    rd = reliability_diagram(yte, p, n_bins=10)
    print(rd.show_text())

    print("\n=== TILT CLASSIFIER (5-fold CV) ===")

    print("\n[1] Logistic regression, baseline:")
    rep = cross_validate(TiltClassifier, Xt, yt, k=5,
                         fit_kwargs={"epochs": 400})
    print(rep.show_text())

    print("\n[2] MLP (16 hidden units):")
    rep_mlp_t = cross_validate(lambda: MLPClassifier(hidden_dim=16), Xt, yt, k=5,
                               fit_kwargs={"epochs": 800, "lr": 0.05})
    print(rep_mlp_t.show_text())


if __name__ == "__main__":
    main()
