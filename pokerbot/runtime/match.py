"""Match-play evaluator with confidence intervals.

Why we need this even though we have analytical EV computation:

  - Once we move past Kuhn (where exact tree-walk EV is feasible),
    the only practical way to compare two strategies is to play
    a lot of hands and average.
  - For Leduc, the tree is small enough for exact EV, but we want
    to validate that match-play numbers MATCH the analytical numbers.
    If they don't, something is broken.
  - For NLHE later, this will be the primary evaluation tool.

What we report:
  - mean(p0_chips_per_hand)
  - 95% confidence interval (via the standard error of the mean)
  - mbb/g — milli-big-blinds per game, the standard poker-bench unit

Two ways to play:
  1. Two HeuristicPlayers (synthetic, fast).
  2. A Strategy dict from CFR + a HeuristicPlayer.
     Internally, we wrap the strategy as a 'StrategyPlayer' that samples
     actions according to the dict's per-info-set distributions.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from pokerbot.games.leduc import LeducPoker
from pokerbot.runtime.heuristic_player import LeducHeuristicPlayer
from pokerbot.runtime.session import _play_one_hand


class StrategyPlayer:
    """Adapter: a CFR/CFR+ strategy dict acts like a Leduc player."""

    def __init__(self, strategy: dict, rng: random.Random | None = None, fallback_uniform: bool = True):
        self.strategy = strategy
        self.rng = rng or random.Random()
        self.fallback_uniform = fallback_uniform

    def decide(self, game, state, legal_actions, my_player: int) -> str:
        info_set = game.info_set_key(state, my_player)
        dist = self.strategy.get(info_set)
        if dist is None:
            if self.fallback_uniform:
                return self.rng.choice(legal_actions)
            raise KeyError(f"strategy missing info set {info_set}")
        # Restrict and renormalize to legal_actions (in case strategy has stale keys).
        items = [(a, dist.get(a, 0.0)) for a in legal_actions]
        total = sum(p for _, p in items)
        if total <= 0:
            return self.rng.choice(legal_actions)
        r = self.rng.random() * total
        cum = 0.0
        for a, p in items:
            cum += p
            if r < cum:
                return a
        return items[-1][0]

    # No-op observation hook so it duck-types with HeuristicPlayer.
    def observe_result(self, net_chips: float) -> None:
        pass


@dataclass
class MatchResult:
    n_hands: int
    p0_mean: float
    p0_stderr: float
    p0_ci_low: float
    p0_ci_high: float

    @property
    def mbb_per_game(self) -> float:
        # 1 chip in Leduc round-2 bet = 4 chips; treat 4 as the "big bet" unit;
        # then *1000 mbb. So mbb/g = mean / 4 * 1000.
        return self.p0_mean / 4.0 * 1000.0

    def __str__(self) -> str:
        return (
            f"P0 EV/hand = {self.p0_mean:+.4f} chips  "
            f"(95% CI [{self.p0_ci_low:+.4f}, {self.p0_ci_high:+.4f}])  "
            f"= {self.mbb_per_game:+.1f} mbb/g over {self.n_hands} hands"
        )


def play_match(
    p0,
    p1,
    n_hands: int,
    rng: random.Random | None = None,
) -> MatchResult:
    """Play n_hands of Leduc and return summary stats with 95% CI."""
    rng = rng or random.Random()
    game = LeducPoker()

    p0_chips: list = []
    for _ in range(n_hands):
        trace = _play_one_hand(game, p0, p1, rng)
        # If a player exposes observe_showdown (e.g. AdaptiveBotPlayer), tell
        # them the opponent's card so they can label past bets.
        if trace.went_to_showdown and trace.board_card is not None:
            opp_card_for_p0 = trace.hole_cards[1]
            opp_card_for_p1 = trace.hole_cards[0]
            if hasattr(p0, "observe_showdown"):
                p0.observe_showdown(opp_card_for_p0, trace.board_card)
            if hasattr(p1, "observe_showdown"):
                p1.observe_showdown(opp_card_for_p1, trace.board_card)
        p0.observe_result(trace.p0_net)
        p1.observe_result(trace.p1_net)
        p0_chips.append(trace.p0_net)

    n = len(p0_chips)
    mean = sum(p0_chips) / n
    var = sum((x - mean) ** 2 for x in p0_chips) / max(1, n - 1)
    stderr = math.sqrt(var / n)
    z = 1.96
    return MatchResult(
        n_hands=n,
        p0_mean=mean,
        p0_stderr=stderr,
        p0_ci_low=mean - z * stderr,
        p0_ci_high=mean + z * stderr,
    )
