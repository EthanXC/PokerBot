"""Single source of truth for bluff-feature construction.

Both the offline trace-based extractor (runtime/features.py) and the live
AdaptiveBotPlayer were building the same 12-feature bluff vector with
slightly drifting code. This module centralizes it.

Public API:
    build_bluff_feature_vector(ctx) -> 1-D numpy array of length 12

Where ctx is a small dataclass capturing everything we know at decision time:
the betting action, the board card, the bettor's running stats, etc.

Why a dataclass for the context: it makes online and offline code call the
SAME function with the SAME semantics; if a feature definition changes,
both code paths get the update for free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerbot.games.leduc import card_rank


@dataclass
class BluffFeatureContext:
    # In-the-moment.
    bet_action: str            # 'b' or 'r'
    board_card: int | None     # community card or None
    n_bets_this_street: int    # already-issued bets/raises this round (BEFORE this one)
    decisions_this_street: int # # of acts taken this round so far (BEFORE this one)
    bettor_position: int       # 0 or 1 (P1 has position in Leduc)
    street: int                # 1 or 2
    round_history_before: str  # round-history string up to (not incl) this action

    # Bettor's running stats.
    bettor_aggression_count: int    # # of bets/raises lifetime
    bettor_decision_count: int      # # of total decisions lifetime
    bettor_bluff_count: int         # # of confirmed bluffs at showdown
    bettor_value_count: int         # # of confirmed value bets at showdown
    bettor_showdown_count: int      # # of hands going to showdown
    bettor_hand_count: int          # # of hands seen
    bettor_actions_this_hand: int = 0   # bettor's b/r count earlier in this hand


def _board_strength(board_card) -> float:
    if board_card is None:
        return 0.0
    return [0.3, 0.5, 0.7][card_rank(board_card)]


def build_bluff_feature_vector(ctx: BluffFeatureContext) -> np.ndarray:
    """Construct the 12-feature bluff vector from a decision context."""
    bet_size_ratio = 1.5 if ctx.bet_action == "r" else 1.0
    overbet = 1.0 if ctx.bet_action == "r" else 0.0
    donk_lead = 1.0 if (
        ctx.street == 2
        and ctx.bettor_position == 0
        and ctx.round_history_before == ""
    ) else 0.0
    pot_committed = min(1.0, ctx.n_bets_this_street * 0.15 + 0.1)
    prior_aggression = (
        (ctx.bettor_aggression_count + 2.0)
        / (ctx.bettor_decision_count + 7.0)
    )
    bettor_recent_bluff_rate = (
        (ctx.bettor_bluff_count + 1.0)
        / (ctx.bettor_bluff_count + ctx.bettor_value_count + 2.0)
    )
    bettor_showdown_rate = (
        (ctx.bettor_showdown_count + 1.0) / (ctx.bettor_hand_count + 2.0)
    )
    street_pace = ctx.n_bets_this_street / max(1, ctx.decisions_this_street)

    return np.array([
        bet_size_ratio,
        _board_strength(ctx.board_card),
        prior_aggression,
        float(ctx.bettor_position),
        pot_committed,
        ctx.n_bets_this_street + 1,
        overbet,
        donk_lead,
        bettor_recent_bluff_rate,
        bettor_showdown_rate,
        float(ctx.bettor_actions_this_hand),
        street_pace,
    ], dtype=float)
