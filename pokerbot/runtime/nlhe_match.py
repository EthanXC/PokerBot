"""N-player NLHE match harness.

Runs N hands at a table of K players, rotating the button each hand.
Tracks per-seat chip totals with confidence intervals.

Each player object must support:
    .decide(game, state, legal_actions, my_player) -> action
    .observe_result(net_chips) -> None         (optional)
    .observe_showdown(...)                     (optional, for online learning)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from pokerbot.core.cards import Card, VALID_RANKS, VALID_SUITS
from pokerbot.games.nlhe import NLHE, NLHEConfig


@dataclass
class TableMatchResult:
    """Per-seat summary across all hands at the table."""
    n_hands: int
    seat_means: list = field(default_factory=list)
    seat_stderrs: list = field(default_factory=list)
    seat_ci_low: list = field(default_factory=list)
    seat_ci_high: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"  Match results over {self.n_hands} hands:"]
        for i, (m, lo, hi) in enumerate(
            zip(self.seat_means, self.seat_ci_low, self.seat_ci_high)
        ):
            mbb_per_g = m * 1000 / 2  # 1 BB = 2 chips at default config
            lines.append(
                f"    seat {i}: {m:+.4f} chips/hand  "
                f"CI [{lo:+.4f}, {hi:+.4f}]  ({mbb_per_g:+.0f} mbb/g)"
            )
        return "\n".join(lines)


def _shuffle_deck(rng: random.Random) -> list:
    deck = [Card(r, s) for r in VALID_RANKS for s in VALID_SUITS]
    rng.shuffle(deck)
    return deck


def play_table(
    players: list,
    n_hands: int,
    config: NLHEConfig | None = None,
    rng: random.Random | None = None,
) -> TableMatchResult:
    """Play n_hands at a K-player NLHE table; return per-seat summary."""
    if config is None:
        config = NLHEConfig(n_players=len(players))
    if config.n_players != len(players):
        raise ValueError(f"config.n_players={config.n_players} != len(players)={len(players)}")

    rng = rng or random.Random()
    game = NLHE(config)
    n = config.n_players
    seat_chip_lists = [[] for _ in range(n)]

    for hand_idx in range(n_hands):
        deck = _shuffle_deck(rng)
        button = hand_idx % n
        state = game.initial_state(deck_order=deck, button=button)

        # Walk the hand.
        steps = 0
        while not game.is_terminal(state) and steps < 500:
            if game.is_chance(state):
                outcomes = game.chance_outcomes(state)
                # Sample by probability.
                r = rng.random()
                cum = 0.0
                chosen = outcomes[-1][0]
                for a, p in outcomes:
                    cum += p
                    if r < cum:
                        chosen = a
                        break
                state = game.apply(state, chosen)
            else:
                seat = state.actor
                player = players[seat]
                legal = game.legal_actions(state)
                action = player.decide(game, state, legal, seat)
                if action not in legal:
                    # Illegal action — fallback to first legal.
                    action = legal[0]
                state = game.apply(state, action)
            steps += 1

        # Hand complete; collect utilities.
        all_nets = [game.utility(state, seat) for seat in range(n)]
        for seat in range(n):
            seat_chip_lists[seat].append(all_nets[seat])
            if hasattr(players[seat], "observe_result"):
                players[seat].observe_result(all_nets[seat])
            # Multi-opponent bots want the FULL hand context to update per-seat
            # stats; pass it through if they support the hook.
            if hasattr(players[seat], "observe_hand_finalized"):
                players[seat].observe_hand_finalized(state, seat, all_nets)

    # Aggregate stats.
    result = TableMatchResult(n_hands=n_hands)
    for seat_chips in seat_chip_lists:
        m = sum(seat_chips) / max(1, len(seat_chips))
        var = sum((x - m) ** 2 for x in seat_chips) / max(1, len(seat_chips) - 1)
        se = math.sqrt(var / max(1, len(seat_chips)))
        z = 1.96
        result.seat_means.append(m)
        result.seat_stderrs.append(se)
        result.seat_ci_low.append(m - z * se)
        result.seat_ci_high.append(m + z * se)
    return result
