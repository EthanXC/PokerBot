"""Tests for the EM-fitted GMM.

Validates:
  - Log-likelihood is non-decreasing across EM iterations.
  - With well-separated synthetic clusters, GMM recovers the true means
    (up to permutation of components).
  - Responsibilities are valid probability distributions.
"""
from __future__ import annotations

import unittest

import numpy as np

from pokerbot.learning.gmm import fit_gmm


def synthesize_3_clusters(n_per_cluster: int, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    means = np.array([[0, 0], [5, 5], [-5, 4]], dtype=float)
    samples = []
    labels = []
    for k, mu in enumerate(means):
        s = rng.normal(loc=mu, scale=0.5, size=(n_per_cluster, 2))
        samples.append(s)
        labels.extend([k] * n_per_cluster)
    X = np.vstack(samples)
    y = np.array(labels)
    return X, y, means


class GMMTest(unittest.TestCase):
    def test_loglikelihood_monotone_nondecreasing(self):
        X, _, _ = synthesize_3_clusters(100)
        gmm = fit_gmm(X, n_components=3, seed=0)
        ll = gmm.log_likelihood_history
        # EM theory: log-likelihood non-decreasing every iteration.
        for i in range(1, len(ll)):
            self.assertGreaterEqual(
                ll[i], ll[i - 1] - 1e-6,
                f"log-likelihood decreased at iter {i}: {ll[i-1]} -> {ll[i]}"
            )

    def test_recovers_known_means(self):
        X, _, true_means = synthesize_3_clusters(200, seed=42)
        gmm = fit_gmm(X, n_components=3, seed=42)
        # Match each fitted mean to its closest true mean.
        # Total assignment cost (sum of distances) should be small.
        used = set()
        total_dist = 0.0
        for tm in true_means:
            dists = [(i, np.linalg.norm(tm - gmm.means[i])) for i in range(3) if i not in used]
            dists.sort(key=lambda kv: kv[1])
            best_i, best_d = dists[0]
            used.add(best_i)
            total_dist += best_d
        self.assertLess(total_dist, 1.5,
                        f"GMM means too far from truth, total_dist={total_dist}")

    def test_responsibilities_sum_to_one(self):
        X, _, _ = synthesize_3_clusters(50)
        gmm = fit_gmm(X, n_components=3, seed=0)
        R = gmm.responsibilities(X)
        sums = R.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)

    def test_predict_assigns_correct_clusters(self):
        X, y_true, _ = synthesize_3_clusters(150, seed=7)
        gmm = fit_gmm(X, n_components=3, seed=7)
        y_pred = gmm.predict(X)

        # Greedy permutation: for each predicted label, find the true label
        # that maps to it most often. Then accuracy should be ~1.
        n_correct = 0
        for k in range(3):
            mask = (y_pred == k)
            if not mask.any():
                continue
            true_labels_in_cluster = y_true[mask]
            modal = int(np.bincount(true_labels_in_cluster).argmax())
            n_correct += (true_labels_in_cluster == modal).sum()
        accuracy = n_correct / len(X)
        self.assertGreater(accuracy, 0.95, f"clustering accuracy too low: {accuracy}")

    def test_two_overlapping_clusters_still_runs(self):
        rng = np.random.default_rng(1)
        X = np.vstack([
            rng.normal(0, 1, size=(100, 2)),
            rng.normal(1, 1, size=(100, 2)),
        ])
        gmm = fit_gmm(X, n_components=2, seed=1, max_iters=50)
        # Doesn't have to do well; just shouldn't crash and LL should improve.
        self.assertGreater(
            gmm.log_likelihood_history[-1],
            gmm.log_likelihood_history[0] - 1e-6
        )


if __name__ == "__main__":
    unittest.main()
