"""MLE / MAP opponent-strategy estimator.

Course topics covered: Probabilistic Reasoning, MLE, Bayesian inference.

PROBLEM
-------
We observe the opponent making decisions. At each information set I they
take some action a. We want to estimate their strategy pi_opp(a | I).

If we treat each info set independently and assume the opponent's strategy
is a (fixed but unknown) categorical distribution at each I, then:

  - The MLE for pi_opp(a | I) is just the empirical frequency:
        p_hat[a | I] = count(a, I) / sum_a' count(a', I)

  - With a Dirichlet(alpha) prior, the MAP / posterior-mean estimate is:
        p_hat[a | I] = (alpha[a] + count(a, I)) / (sum_a' alpha[a'] + sum_a' count(a', I))

This is conjugate prior magic: the posterior over a categorical with a
Dirichlet prior is also Dirichlet (just add the counts to the prior alphas).

Why MAP/MAP > raw MLE for our use case:
  - Empirical frequency is unstable with few observations.
  - The Dirichlet prior shrinks small-sample estimates toward uniform,
    so we don't over-react to opponent behavior we've barely seen.

API:
    mle = StrategyMLE(prior_alpha=1.0)
    mle.observe(info_set="K:b", action="b")
    mle.observe(info_set="K:b", action="b")
    mle.observe(info_set="K:b", action="p")
    print(mle.strategy())
    # -> {"K:b": {"b": 0.6, "p": 0.4}}  (with prior_alpha=1.0)

The estimator slots directly into the HybridPolicy: feed an opponent's
modeled strategy into update_opponent_model().
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable


class StrategyMLE:
    """Per-info-set Dirichlet-Categorical Bayesian estimator.

    Each info set has its own posterior over actions; observations at
    one info set don't influence estimates at others.

    Attributes:
        prior_alpha: Dirichlet concentration. Higher = stronger pull
            toward uniform with little data.
    """

    def __init__(self, prior_alpha: float = 1.0):
        if prior_alpha <= 0:
            raise ValueError("prior_alpha must be positive")
        self.prior_alpha = prior_alpha
        # info_set -> action -> count
        self._counts: dict = defaultdict(lambda: defaultdict(int))
        # info_set -> set of actions ever seen (so unseen actions still
        # show up in the strategy with prior weight)
        self._seen_actions: dict = defaultdict(set)

    def observe(self, info_set: Hashable, action: Hashable) -> None:
        self._counts[info_set][action] += 1
        self._seen_actions[info_set].add(action)

    def observe_legal_actions(self, info_set: Hashable, actions) -> None:
        """Tell the estimator which actions are legal at an info set so
        that unobserved actions still get prior probability.

        Without this, an action we've never seen the opponent take won't
        appear in strategy() output.
        """
        self._seen_actions[info_set].update(actions)

    def n_observations(self, info_set: Hashable) -> int:
        return sum(self._counts[info_set].values())

    def posterior_mean(self, info_set: Hashable) -> dict:
        """Return {action: probability} = posterior mean of Dirichlet.

        Equivalent to MAP for Dirichlet+Categorical. Falls back to uniform
        over self._seen_actions[info_set] when there's no data.
        """
        actions = sorted(self._seen_actions[info_set])
        if not actions:
            return {}
        counts = self._counts[info_set]
        total = sum(counts.values()) + self.prior_alpha * len(actions)
        return {a: (counts.get(a, 0) + self.prior_alpha) / total for a in actions}

    def mle(self, info_set: Hashable) -> dict:
        """Pure MLE (no prior). Returns uniform if no data."""
        actions = sorted(self._seen_actions[info_set])
        if not actions:
            return {}
        counts = self._counts[info_set]
        total = sum(counts.values())
        if total == 0:
            return {a: 1.0 / len(actions) for a in actions}
        return {a: counts.get(a, 0) / total for a in actions}

    def strategy(self, use_map: bool = True) -> dict:
        """Return the full {info_set: {action: prob}} estimate."""
        if use_map:
            return {I: self.posterior_mean(I) for I in self._seen_actions}
        return {I: self.mle(I) for I in self._seen_actions}

    def confidence(self, info_set: Hashable) -> float:
        """How much weight does data have vs. prior at this info set?

        Returns 0 with no data, → 1 with many observations.
        """
        n = self.n_observations(info_set)
        actions = self._seen_actions[info_set]
        prior_mass = self.prior_alpha * len(actions)
        return n / (n + prior_mass) if (n + prior_mass) > 0 else 0.0

    def overall_confidence(self) -> float:
        """Average confidence across all observed info sets, weighted by
        observations. A single scalar to plug into HybridPolicy.
        """
        if not self._seen_actions:
            return 0.0
        total_n = 0
        total_weighted = 0.0
        for I in self._seen_actions:
            n = self.n_observations(I)
            c = self.confidence(I)
            total_n += n
            total_weighted += c * n
        return total_weighted / total_n if total_n > 0 else 0.0
