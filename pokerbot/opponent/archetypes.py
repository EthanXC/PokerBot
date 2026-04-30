"""Bayesian archetype model.

Maintains a posterior over a small set of opponent types given observed HUD
stats. Each archetype is a soft template — a probability distribution over
where each stat "ought to be" if the player were that type.

Likelihood model: each stat is treated as Gaussian around the archetype's
mean with a fixed std. Independence assumption across stats (false but useful
and dramatically simpler than a joint model).

Posterior update:
    P(type | stats) ∝ P(type) * Π_s N(stat_s; μ_type_s, σ_type_s)

Returned posterior is normalized. The closer one archetype's means are to
the observed stats, the higher its posterior weight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from pokerbot.opponent.stats import OpponentStats


@dataclass
class Archetype:
    name: str
    # Each (stat_name, mean, std) — std reflects how tolerant we are.
    means: dict = field(default_factory=dict)
    stds: dict = field(default_factory=dict)
    prior: float = 1.0


def _arch(name: str, prior: float = 1.0, **kwargs) -> Archetype:
    """Build an archetype from kwargs of (stat_name=mean) and a default std=0.08."""
    means = {}
    stds = {}
    for key, value in kwargs.items():
        if isinstance(value, tuple):
            mean, std = value
        else:
            mean, std = value, 0.08
        means[key] = mean
        stds[key] = std
    return Archetype(name=name, means=means, stds=stds, prior=prior)


# Numbers below are loose stereotypes from the poker training literature
# (e.g. Jonathan Little, PokerTracker leak finder defaults). They're priors;
# the bot updates against actual play.
ARCHETYPES: list[Archetype] = [
    _arch(
        "TAG",   # tight-aggressive — solid winner
        VPIP=0.22, PFR=0.18, AGG=0.55, F3B=0.55, WTSD=0.27, prior=1.0,
    ),
    _arch(
        "LAG",   # loose-aggressive — tough, lots of bluffs
        VPIP=0.32, PFR=0.27, AGG=0.65, F3B=0.40, WTSD=0.28, prior=0.6,
    ),
    _arch(
        "fish",  # loose-passive calling station
        VPIP=0.45, PFR=0.10, AGG=0.30, F3B=0.30, WTSD=0.40, prior=1.5,
    ),
    _arch(
        "nit",   # super tight
        VPIP=0.12, PFR=0.10, AGG=0.45, F3B=0.75, WTSD=0.20, prior=0.8,
    ),
    _arch(
        "maniac",  # bets/raises everything
        VPIP=0.55, PFR=0.45, AGG=0.80, F3B=0.20, WTSD=0.35, prior=0.3,
    ),
    _arch(
        "balanced",  # close to GTO solver baselines
        VPIP=0.25, PFR=0.20, AGG=0.55, F3B=0.55, WTSD=0.27, prior=0.8,
    ),
]


class ArchetypeModel:
    """Posterior over archetypes given observed stats."""

    def __init__(self, archetypes: Iterable[Archetype] = ARCHETYPES):
        self.archetypes = list(archetypes)
        # Cache prior log-probs.
        prior_sum = sum(a.prior for a in self.archetypes)
        self._log_priors = {
            a.name: math.log(a.prior / prior_sum) for a in self.archetypes
        }

    def posterior(self, stats: OpponentStats) -> dict[str, float]:
        """Return a dict {archetype_name: probability} normalized to 1."""
        # Effective number of hands gates how much the likelihood can influence
        # the prior. With <30 hands of data the posterior is basically the prior.
        confidence = min(1.0, stats.vpip.n / 100.0)

        log_post: dict[str, float] = {}
        for arch in self.archetypes:
            ll = 0.0
            for stat_name, mean in arch.means.items():
                std = arch.stds.get(stat_name, 0.08)
                observed = self._get_stat(stats, stat_name)
                if observed is None:
                    continue
                # Gaussian log-likelihood.
                z = (observed - mean) / std
                ll += -0.5 * z * z
            # Down-weight likelihood by confidence (with no data, log_likelihood -> 0).
            ll *= confidence
            log_post[arch.name] = self._log_priors[arch.name] + ll

        # Normalize via log-sum-exp.
        m = max(log_post.values())
        unnorm = {k: math.exp(v - m) for k, v in log_post.items()}
        total = sum(unnorm.values())
        return {k: v / total for k, v in unnorm.items()}

    def most_likely(self, stats: OpponentStats) -> tuple[str, float]:
        post = self.posterior(stats)
        name, p = max(post.items(), key=lambda kv: kv[1])
        return name, p

    @staticmethod
    def _get_stat(stats: OpponentStats, name: str) -> float | None:
        mapping = {
            "VPIP": stats.vpip,
            "PFR": stats.pfr,
            "3BET": stats.three_bet,
            "F3B": stats.fold_to_3bet,
            "CBET": stats.cbet,
            "F2C": stats.fold_to_cbet,
            "WTSD": stats.wtsd,
            "AGG": stats.aggression,
        }
        s = mapping.get(name)
        if s is None or s.n == 0:
            return None
        return s.estimate
