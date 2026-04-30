"""Heuristic NLHE players, multi-street.

Each player is parameterized by a behavioral profile. The hand-strength
estimator uses our existing 7-card evaluator + a simple lookahead:

  - Preflop: equity bucket from hand category (pairs, suited connectors, etc.).
  - Postflop: hand strength = our hand's score class divided by max class.

This is not solver-quality, but it gives us realistic, distinguishable
opponent behaviors to test the bot against.

Profiles share the same interface as the Leduc heuristic player:
    .observe_result(net_chips)   — informs tilt mechanic
    .decide(game, state, legal_actions, my_player) -> action
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from pokerbot.core.cards import Card, RANK_TO_VALUE
from pokerbot.core.evaluator import HandEvaluator
from pokerbot.games.nlhe import (
    NLHE, NLHEState, NLHEConfig,
    FOLD, CHECK, CALL, POT_BET, HALF_POT_BET, ALL_IN,
)


# --- Hand strength heuristic ---

def preflop_strength(hole: tuple) -> float:
    """Return a 0..1 strength score for a starting hand."""
    a, b = hole
    r1 = RANK_TO_VALUE[a.rank]
    r2 = RANK_TO_VALUE[b.rank]
    suited = a.suit == b.suit
    pair = (r1 == r2)
    high = max(r1, r2)
    low = min(r1, r2)
    gap = high - low

    if pair:
        # Pairs scale from 0.45 (22) up to 0.95 (AA)
        return min(0.95, 0.45 + (r1 - 2) * (0.5 / 12))
    base = (high * 0.05) + (low * 0.02)         # high cards matter most
    if suited:
        base += 0.05
    if gap == 1:
        base += 0.04                            # connectors
    elif gap == 2:
        base += 0.02
    elif gap >= 5:
        base -= 0.05                            # disconnected
    return max(0.05, min(0.85, base))


def postflop_strength(hole: tuple, board: tuple) -> float:
    """0..1 score for our 5..7 card hand.

    Differentiates top pair / overpair / middle pair / bottom pair so the
    bucketing reflects real equity differences. Top pair on a King-high board
    deserves bucket 2-3 (medium-strong), not bucket 1 along with bottom pair.
    """
    if len(board) < 3:
        return preflop_strength(hole)
    cards = list(hole) + list(board)
    if len(cards) == 5:
        score = HandEvaluator.score_five(cards)
    elif len(cards) == 6:
        from itertools import combinations
        score = max(HandEvaluator.score_five(list(c)) for c in combinations(cards, 5))
    else:
        score = HandEvaluator.score_seven(cards)
    cat = score[0]
    cat_to_strength = {
        0: 0.10,  # high card
        1: 0.35,  # pair (baseline; we'll modify by pair position vs board)
        2: 0.62,  # two pair
        3: 0.72,  # trips
        4: 0.78,  # straight
        5: 0.83,  # flush
        6: 0.90,  # full house
        7: 0.95,  # quads
        8: 0.99,  # straight flush
    }
    base = cat_to_strength[cat]

    # For PAIRS specifically, distinguish top/over/middle/bottom pair.
    if cat == 1 and len(score) >= 2:
        pair_rank = score[1]
        board_ranks = sorted(
            [RANK_TO_VALUE[c.rank] for c in board], reverse=True
        )
        top_board = board_ranks[0]
        # Overpair: pocket pair higher than the board top.
        # Top pair: matches top board card.
        # Middle pair: matches a non-top board card.
        # Bottom pair: matches the lowest board card.
        # Pocket pair below board top = "underpair" (weakish).
        hole_ranks = [RANK_TO_VALUE[c.rank] for c in hole]
        is_pocket_pair = hole_ranks[0] == hole_ranks[1]
        if is_pocket_pair and pair_rank > top_board:
            base = 0.65   # overpair
        elif pair_rank == top_board:
            base = 0.55   # top pair
            # Bump for stronger kicker
            if len(score) >= 3:
                kicker = score[2]
                base += max(0.0, (kicker - 10) * 0.01)
        elif pair_rank in board_ranks[1:]:
            # Pair matches a non-top board card → middle/bottom pair
            position_among_board = board_ranks.index(pair_rank)
            if position_among_board == 1:
                base = 0.35   # middle pair
            else:
                base = 0.25   # bottom pair
        elif is_pocket_pair:
            base = 0.30   # underpair
        # Else: shouldn't really happen if we have a pair; fall through

    elif cat == 2 and len(score) >= 2:
        # Two-pair bump for top two
        top_pair = score[1]
        board_top = max(RANK_TO_VALUE[c.rank] for c in board)
        if top_pair >= board_top:
            base += 0.05

    elif cat == 3 and len(score) >= 2:
        # Trips bump
        base += 0.03

    return max(0.05, min(0.99, base))


# --- Profiles ---

@dataclass
class NLHEProfile:
    name: str
    open_threshold: float       # min preflop strength to open-raise
    call_threshold: float       # min preflop strength to call a raise
    bluff_freq: float           # P(bluff with weak hand)
    cbet_freq: float            # P(continuation bet on flop after preflop raise)
    value_threshold: float      # min postflop strength to value-bet
    fold_threshold: float       # below this strength, fold to a bet
    raise_freq_when_strong: float
    tilt_susceptibility: float = 0.2


PROFILES = {
    "tight_passive": NLHEProfile(
        name="tight_passive",
        open_threshold=0.65, call_threshold=0.55,
        bluff_freq=0.05, cbet_freq=0.5,
        value_threshold=0.55, fold_threshold=0.45,
        raise_freq_when_strong=0.30, tilt_susceptibility=0.1,
    ),
    "loose_aggressive": NLHEProfile(
        name="loose_aggressive",
        open_threshold=0.40, call_threshold=0.30,
        bluff_freq=0.35, cbet_freq=0.75,
        value_threshold=0.50, fold_threshold=0.30,
        raise_freq_when_strong=0.55, tilt_susceptibility=0.2,
    ),
    "calling_station": NLHEProfile(
        name="calling_station",
        open_threshold=0.55, call_threshold=0.25,
        bluff_freq=0.05, cbet_freq=0.45,
        value_threshold=0.40, fold_threshold=0.20,
        raise_freq_when_strong=0.10, tilt_susceptibility=0.1,
    ),
    "maniac": NLHEProfile(
        name="maniac",
        open_threshold=0.30, call_threshold=0.20,
        bluff_freq=0.55, cbet_freq=0.85,
        value_threshold=0.45, fold_threshold=0.15,
        raise_freq_when_strong=0.70, tilt_susceptibility=0.4,
    ),
    "tilt_prone": NLHEProfile(
        name="tilt_prone",
        open_threshold=0.55, call_threshold=0.40,
        bluff_freq=0.20, cbet_freq=0.60,
        value_threshold=0.50, fold_threshold=0.35,
        raise_freq_when_strong=0.40, tilt_susceptibility=0.6,
    ),
}


class NLHEHeuristicPlayer:
    """A profile-driven NLHE player."""

    def __init__(self, profile: NLHEProfile, rng: random.Random | None = None):
        self.profile = profile
        self.rng = rng or random.Random()
        self._recent_results: deque = deque(maxlen=5)

    def observe_result(self, net_chips: float) -> None:
        self._recent_results.append(net_chips)

    def is_tilted(self) -> bool:
        if len(self._recent_results) < 3:
            return False
        return sum(self._recent_results) < -10.0  # NLHE scale: $10+ recent loss

    def effective_profile(self) -> NLHEProfile:
        if not self.is_tilted():
            return self.profile
        s = self.profile.tilt_susceptibility
        # Tilt: lower fold threshold, higher bluff freq, looser calling
        return NLHEProfile(
            name=self.profile.name + "*tilted",
            open_threshold=max(0.0, self.profile.open_threshold - 0.15 * s),
            call_threshold=max(0.0, self.profile.call_threshold - 0.20 * s),
            bluff_freq=min(1.0, self.profile.bluff_freq * (1 + 1.5 * s)),
            cbet_freq=self.profile.cbet_freq,
            value_threshold=self.profile.value_threshold,
            fold_threshold=max(0.0, self.profile.fold_threshold - 0.15 * s),
            raise_freq_when_strong=self.profile.raise_freq_when_strong,
            tilt_susceptibility=s,
        )

    def decide(self, game: NLHE, state: NLHEState, legal_actions: list, my_player: int) -> str:
        prof = self.effective_profile()
        my_hole = state.hole_cards[my_player]
        if state.round_idx == 0:
            strength = preflop_strength(my_hole)
        else:
            strength = postflop_strength(my_hole, state.board)

        owed = state.bet_to_match - state.contributed_this_round[my_player]
        facing_bet = owed > 0

        # Add some bluff randomness
        is_bluffing = self.rng.random() < prof.bluff_freq
        effective_strength = max(strength, 0.85) if (is_bluffing and not facing_bet) else strength

        if facing_bet:
            # Decide fold / call / raise
            if effective_strength < prof.fold_threshold and FOLD in legal_actions:
                return FOLD
            if effective_strength > 0.85 and self.rng.random() < prof.raise_freq_when_strong:
                if POT_BET in legal_actions:
                    return POT_BET
                if ALL_IN in legal_actions:
                    return ALL_IN
            if CALL in legal_actions:
                return CALL
            if FOLD in legal_actions:
                return FOLD
            return legal_actions[0]
        else:
            # No bet to call. Decide check / bet.
            # Preflop unraised pot in 3+ handed: check is rare (we should at least call/raise BB)
            if state.round_idx == 0 and effective_strength >= prof.open_threshold:
                if POT_BET in legal_actions:
                    return POT_BET
            # Postflop c-bet logic
            if state.round_idx >= 1 and self.rng.random() < prof.cbet_freq and effective_strength >= prof.value_threshold * 0.7:
                if POT_BET in legal_actions:
                    return POT_BET
            if effective_strength > 0.85 and self.rng.random() < prof.raise_freq_when_strong:
                if POT_BET in legal_actions:
                    return POT_BET
            if CHECK in legal_actions:
                return CHECK
            if CALL in legal_actions:
                return CALL
            return legal_actions[0]


def make_nlhe_player(profile_name: str, rng: random.Random | None = None) -> NLHEHeuristicPlayer:
    return NLHEHeuristicPlayer(PROFILES[profile_name], rng=rng)
