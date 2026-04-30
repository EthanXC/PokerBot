"""Gaussian Mixture Model fit by Expectation-Maximization.

Course topics: Expectation Maximization, Gaussian Mixture Models.

What it does for the bot
------------------------
The hand-coded archetypes in `pokerbot.opponent.archetypes` are guesses about
where TAG/LAG/fish/nit live in stat-space. With a dataset of player stat
vectors, we can LEARN the cluster centers from data. EM-fitted GMM finds:

  - K cluster means mu_k in stat space (the "learned archetypes")
  - cluster covariances Sigma_k (how spread each archetype is)
  - cluster weights pi_k (how common each archetype is in the population)

Then for any new opponent's stat vector x, the responsibility:

    P(z = k | x) = pi_k * N(x; mu_k, Sigma_k) / sum_j pi_j * N(x; mu_j, Sigma_j)

is exactly the Bayesian posterior over which archetype this opponent matches.

Implementation
--------------
Diagonal-covariance GMM (faster, fewer parameters). Standard EM:

  E-step: compute responsibilities r[n, k] = P(z = k | x_n)
  M-step: update pi_k, mu_k, sigma_k as weighted MLE under r[n, k]

We run EM until log-likelihood improvement is below a threshold or we hit
the max iterations. K-means++ initialization to avoid pathological starts.

Using numpy for the linear algebra; the algorithm itself is hand-written
so the EM loop is visible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# A small floor for variance to avoid singular covariances ("collapsing"
# components onto a single point).
_VAR_FLOOR = 1e-4


@dataclass
class GaussianMixture:
    """A fitted GMM with diagonal covariances.

    weights: shape (K,)
    means:   shape (K, D)
    vars:    shape (K, D)  -- per-dimension variances
    log_likelihood_history: list of LL after each EM iteration (monotonically
        non-decreasing for a correctly-implemented EM).
    """

    weights: np.ndarray
    means: np.ndarray
    vars: np.ndarray
    log_likelihood_history: list

    @property
    def n_components(self) -> int:
        return len(self.weights)

    def log_prob_per_component(self, X: np.ndarray) -> np.ndarray:
        """log p(x_n | z = k) for each x_n, k. Shape (N, K)."""
        # Diagonal Gaussian log pdf:
        #   log N(x; mu, sigma^2) = -0.5 * sum_d [ (x_d-mu_d)^2 / sigma_d^2 + log(2*pi*sigma_d^2) ]
        # X shape: (N, D); means: (K, D); vars: (K, D)
        N, D = X.shape
        K = self.n_components
        diff = X[:, None, :] - self.means[None, :, :]            # (N, K, D)
        sq = (diff * diff) / self.vars[None, :, :]               # (N, K, D)
        const = np.log(2 * np.pi * self.vars)                    # (K, D)
        return -0.5 * (sq.sum(axis=2) + const.sum(axis=1)[None, :])

    def responsibilities(self, X: np.ndarray) -> np.ndarray:
        """P(z = k | x_n). Shape (N, K), rows sum to 1."""
        log_p = self.log_prob_per_component(X)
        log_w = np.log(self.weights + 1e-300)
        log_joint = log_p + log_w[None, :]                       # (N, K)
        # Normalize per row via log-sum-exp.
        m = log_joint.max(axis=1, keepdims=True)
        unnorm = np.exp(log_joint - m)
        Z = unnorm.sum(axis=1, keepdims=True)
        return unnorm / Z

    def log_likelihood(self, X: np.ndarray) -> float:
        log_p = self.log_prob_per_component(X)
        log_w = np.log(self.weights + 1e-300)
        log_joint = log_p + log_w[None, :]
        # log p(x_n) = logsumexp over components
        m = log_joint.max(axis=1, keepdims=True)
        return float((m.squeeze(1) + np.log(np.exp(log_joint - m).sum(axis=1))).sum())

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard assignment: most likely component for each row."""
        return self.responsibilities(X).argmax(axis=1)


def _kmeans_pp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ seeding: pick first center uniformly; subsequent ones with
    probability proportional to squared distance to nearest existing center.
    This dramatically reduces the chance of bad EM local minima.
    """
    N, D = X.shape
    centers = np.zeros((K, D))
    idx0 = int(rng.integers(N))
    centers[0] = X[idx0]
    for k in range(1, K):
        # Squared dist to nearest center for each point
        diff = X[:, None, :] - centers[None, :k, :]      # (N, k, D)
        d2 = (diff * diff).sum(axis=2).min(axis=1)       # (N,)
        if d2.sum() == 0:
            # All points coincide with existing centers; pick at random
            centers[k] = X[int(rng.integers(N))]
            continue
        probs = d2 / d2.sum()
        idx = int(rng.choice(N, p=probs))
        centers[k] = X[idx]
    return centers


def fit_gmm(
    X,
    n_components: int,
    max_iters: int = 200,
    tol: float = 1e-5,
    seed: int | None = 0,
) -> GaussianMixture:
    """Fit a diagonal-covariance GMM via EM.

    Args:
        X: array-like of shape (N, D)
        n_components: K
        max_iters: maximum EM iterations
        tol: stop when |log_likelihood improvement| < tol
        seed: RNG seed for reproducibility (None to use system entropy)

    Returns:
        GaussianMixture with fitted parameters and log_likelihood_history.
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    K = n_components
    if K < 1:
        raise ValueError("n_components must be >= 1")
    if N < K:
        raise ValueError(f"need at least n_components ({K}) data points, got {N}")

    rng = np.random.default_rng(seed)

    # --- Initialization ---
    means = _kmeans_pp_init(X, K, rng)
    # Initial vars = global per-dim variance, shared across components
    global_var = X.var(axis=0) + _VAR_FLOOR
    vars_ = np.tile(global_var, (K, 1))
    weights = np.ones(K) / K

    model = GaussianMixture(weights=weights, means=means, vars=vars_, log_likelihood_history=[])

    prev_ll = -np.inf
    for it in range(max_iters):
        # --- E-step: responsibilities ---
        R = model.responsibilities(X)                     # (N, K)
        N_k = R.sum(axis=0) + 1e-12                       # (K,)

        # --- M-step ---
        model.weights = N_k / N
        model.means = (R.T @ X) / N_k[:, None]            # (K, D)
        # Weighted variance per component, per dim:
        #   var_kd = sum_n r[n,k] * (x[n,d] - mu_k_d)^2 / N_k
        diff = X[:, None, :] - model.means[None, :, :]    # (N, K, D)
        sq = diff * diff
        model.vars = (R[:, :, None] * sq).sum(axis=0) / N_k[:, None]
        np.maximum(model.vars, _VAR_FLOOR, out=model.vars)

        ll = model.log_likelihood(X)
        model.log_likelihood_history.append(ll)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return model
