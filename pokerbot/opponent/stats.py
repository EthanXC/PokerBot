"""HUD-style stats for an opponent, with empirical-Bayes shrinkage.

The naive way: count "raised preflop" / "saw flop". Tiny samples produce
useless estimates (3/3 = 100% looks like a maniac). So we use a Beta(α, β)
prior and report the posterior mean:

    p_hat = (alpha + successes) / (alpha + beta + opportunities)

For most poker stats the population prior centers somewhere around 0.2-0.3
with effective sample size 5-10. We use Beta(2, 5) by default (mean = 2/7
~= 0.286, ESS = 7).

This is a "shrink toward the prior" mechanism: with 0 hands of data, p_hat
is the prior mean; with thousands of hands, p_hat → empirical frequency.

Stats we track for the human-element analysis:
  - VPIP: voluntarily put money in pot preflop
  - PFR:  preflop raise
  - 3BET: 3-bet preflop
  - F3B:  fold to 3-bet
  - CBET: continuation bet on flop
  - F2C:  fold to flop c-bet
  - WTSD: went to showdown
  - AGG:  aggression frequency post-flop (bet+raise / bet+raise+call)

Plus *temporal* features for tilt detection:
  - recent_loss_chips: chips lost in the last K hands
  - vpip_recent vs vpip_lifetime: change in VPIP after big losses
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class BetaStat:
    """A Beta-prior shrinking estimator for a binary frequency."""

    alpha: float = 2.0   # prior successes
    beta: float = 5.0    # prior failures
    successes: int = 0
    opportunities: int = 0

    def observe(self, success: bool, opportunity: bool = True) -> None:
        if not opportunity:
            return
        self.opportunities += 1
        if success:
            self.successes += 1

    @property
    def estimate(self) -> float:
        return (self.alpha + self.successes) / (
            self.alpha + self.beta + self.opportunities
        )

    @property
    def n(self) -> int:
        return self.opportunities

    @property
    def confidence(self) -> float:
        """Pseudo-confidence in [0, 1]: how much weight has the empirical estimate
        relative to the prior. Returns 0 with no data, → 1 with much data."""
        ess = self.alpha + self.beta
        return self.opportunities / (self.opportunities + ess)


@dataclass
class OpponentStats:
    """Lifetime + recent (windowed) stats for one opponent."""

    name: str = "villain"
    vpip: BetaStat = field(default_factory=lambda: BetaStat(alpha=2, beta=5))
    pfr: BetaStat = field(default_factory=lambda: BetaStat(alpha=1.5, beta=8))
    three_bet: BetaStat = field(default_factory=lambda: BetaStat(alpha=0.6, beta=12))
    fold_to_3bet: BetaStat = field(default_factory=lambda: BetaStat(alpha=4, beta=4))
    cbet: BetaStat = field(default_factory=lambda: BetaStat(alpha=4, beta=4))
    fold_to_cbet: BetaStat = field(default_factory=lambda: BetaStat(alpha=3, beta=4))
    wtsd: BetaStat = field(default_factory=lambda: BetaStat(alpha=2, beta=6))
    aggression: BetaStat = field(default_factory=lambda: BetaStat(alpha=3, beta=5))

    # Tilt detection.
    recent_results: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_vpip: deque = field(default_factory=lambda: deque(maxlen=20))

    hands_played: int = 0

    def observe_hand_result(self, net_chips: float) -> None:
        self.recent_results.append(net_chips)
        self.hands_played += 1

    def observe_voluntary_action(self, voluntary: bool) -> None:
        self.recent_vpip.append(1 if voluntary else 0)

    @property
    def recent_loss(self) -> float:
        if not self.recent_results:
            return 0.0
        return sum(r for r in self.recent_results if r < 0)

    @property
    def recent_loss_severity(self) -> float:
        """Heuristic in [0, 1]. Larger means more recent losses."""
        if not self.recent_results:
            return 0.0
        bad = -min(0.0, self.recent_loss)
        # Normalize by typical-stake scale; using 50 chips as a rough unit.
        return min(1.0, bad / 50.0)

    @property
    def recent_vpip_rate(self) -> float:
        if not self.recent_vpip:
            return self.vpip.estimate
        return sum(self.recent_vpip) / len(self.recent_vpip)

    def tilt_score(self) -> float:
        """A scalar in [0, 1]: how likely is this player on tilt right now?

        Heuristic: weighted combo of (recent_loss_severity, recent_vpip vs lifetime VPIP).
        On tilt, players typically VPIP more after losses.

        Returns 0 if we don't have enough data.
        """
        if self.vpip.n < 20:
            return 0.0
        loss_sev = self.recent_loss_severity
        vpip_jump = max(0.0, self.recent_vpip_rate - self.vpip.estimate)
        # Blend: 60% recent losses, 40% VPIP jump.
        return min(1.0, 0.6 * loss_sev + 0.4 * vpip_jump * 3.0)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "hands": self.hands_played,
            "VPIP": self.vpip.estimate,
            "PFR": self.pfr.estimate,
            "3BET": self.three_bet.estimate,
            "F3B": self.fold_to_3bet.estimate,
            "CBET": self.cbet.estimate,
            "F2C": self.fold_to_cbet.estimate,
            "WTSD": self.wtsd.estimate,
            "AGG": self.aggression.estimate,
            "tilt": self.tilt_score(),
        }
