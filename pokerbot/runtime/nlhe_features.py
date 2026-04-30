"""Extract bluff/tilt training data from NLHE sessions.

This is the NLHE analog of `runtime/features.py` (which works on Leduc).
Auto-labeling at showdown: when a hand reaches showdown, we see every
bettor's hole cards and can label each of their bets as 'bluff' (bottom
30% of equity given board) or 'value' (top 30%).

Returns 12-feature vectors compatible with BluffClassifier.
"""
from __future__ import annotations

import itertools
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pokerbot.core.cards import Card, VALID_RANKS, VALID_SUITS
from pokerbot.core.evaluator import HandEvaluator
from pokerbot.games.nlhe import (
    NLHE, NLHEState, NLHEConfig,
    POT_BET, ALL_IN, FOLD, CALL, CHECK,
)
from pokerbot.runtime.nlhe_player import postflop_strength, preflop_strength
from pokerbot.runtime.nlhe_match import _shuffle_deck


def _hand_strength_label(hole, board) -> int:
    """Returns 1 (bluff) if the bettor has a weak hand, 0 (value) if strong.

    Threshold: postflop_strength < 0.35 → bluff; > 0.55 → value; otherwise drop.
    Returns -1 for "ambiguous, drop this example."
    """
    if not board:
        # Preflop: bluff if weak (J-low, no pair); value if strong (TT+, AK+).
        s = preflop_strength(hole)
    else:
        s = postflop_strength(hole, tuple(board))
    if s < 0.35:
        return 1
    if s > 0.55:
        return 0
    return -1


def _played_one_nlhe_hand(game, players, rng, button: int):
    """Run one hand of NLHE. Returns (final_state, action_log).

    action_log: list of (seat, action, state_at_decision_time)
    """
    deck = _shuffle_deck(rng)
    state = game.initial_state(deck_order=deck, button=button)
    log = []
    steps = 0
    while not game.is_terminal(state) and steps < 500:
        if game.is_chance(state):
            outcomes = game.chance_outcomes(state)
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
            legal = game.legal_actions(state)
            action = players[seat].decide(game, state, legal, seat)
            if action not in legal:
                action = legal[0]
            log.append((seat, action, state))
            state = game.apply(state, action)
        steps += 1
    # Inform players for tilt mechanic.
    for seat in range(game.config.n_players):
        net = game.utility(state, seat)
        if hasattr(players[seat], "observe_result"):
            players[seat].observe_result(net)
    return state, log


def extract_nlhe_bluff_examples(game, players, rng, n_hands: int) -> tuple:
    """Run n_hands and produce (X, y) for BluffClassifier training.

    Only bet/raise actions where the bettor's hand reaches showdown produce
    labeled examples.
    """
    Xs = []
    ys = []
    n = game.config.n_players

    # Per-player rolling stats for feature construction.
    @dataclass
    class _S:
        n_actions: int = 0
        n_aggressive: int = 0
        n_hands: int = 0
        n_showdowns: int = 0
        n_bluffs: int = 0
        n_value: int = 0

    stats = [_S() for _ in range(n)]

    for hand_idx in range(n_hands):
        button = hand_idx % n
        state, log = _played_one_nlhe_hand(game, players, rng, button)

        # Determine who reached showdown (didn't fold).
        showdown_seats = set()
        if "f" not in "".join(a for _, a, _ in log):
            showdown_seats = set(range(n))
        else:
            folded = set()
            for seat, action, _ in log:
                if action == FOLD:
                    folded.add(seat)
            showdown_seats = set(range(n)) - folded

        # Process the action log to extract features for each bet.
        # Track running per-seat aggression as we go (so features match how
        # the bot would compute them online).
        per_seat_actions_so_far = [0] * n
        per_seat_aggressive_so_far = [0] * n
        per_seat_actions_this_hand = [0] * n
        per_seat_aggressive_this_hand = [0] * n

        for (seat, action, st) in log:
            if action in (POT_BET, ALL_IN):
                # Build feature vector at the moment of this bet.
                # Skip if this seat doesn't reach showdown OR doesn't have valid hole/board info.
                if seat not in showdown_seats:
                    per_seat_actions_so_far[seat] += 1
                    per_seat_aggressive_so_far[seat] += 1
                    per_seat_actions_this_hand[seat] += 1
                    per_seat_aggressive_this_hand[seat] += 1
                    continue
                # We need the FINAL board for postflop labeling, since the
                # bettor's effective strength against the showdown board is
                # what determines bluff vs value.
                # For simplicity we use the board AT the bet time for board_str
                # (same as the online bot would see) but the FINAL board for
                # the LABEL.
                final_board = state.board
                board_at_bet = st.board
                hole = st.hole_cards[seat]

                label = _hand_strength_label(hole, final_board)
                if label < 0:
                    per_seat_actions_so_far[seat] += 1
                    per_seat_aggressive_so_far[seat] += 1
                    per_seat_actions_this_hand[seat] += 1
                    per_seat_aggressive_this_hand[seat] += 1
                    continue

                # Build the 12-feature vector (matches BluffClassifier).
                round_h = st.history[st.round_idx] if st.round_idx < len(st.history) else ()
                n_bets_this_street = sum(
                    1 for _, a in round_h if a in (POT_BET, ALL_IN)
                )
                decisions_this_street = len(round_h)

                cum_actions = stats[seat].n_actions + per_seat_actions_so_far[seat]
                cum_aggr = stats[seat].n_aggressive + per_seat_aggressive_so_far[seat]
                prior_aggression = (cum_aggr + 2.0) / (cum_actions + 7.0)

                if board_at_bet:
                    phantom = (Card("2", "C"), Card("3", "D"))
                    used = set(phantom) | set(board_at_bet)
                    if any(c in used for c in phantom):
                        # collision; use 4 of clubs, 5 of diamonds instead
                        phantom = (Card("4", "C"), Card("5", "D"))
                    try:
                        board_str = postflop_strength(phantom, tuple(board_at_bet))
                    except Exception:
                        board_str = 0.5
                else:
                    board_str = 0.0

                pot_committed = min(1.0, st.contributed[seat] / max(1, st.config.starting_stack))
                bluff_rate = (stats[seat].n_bluffs + 1.0) / (stats[seat].n_bluffs + stats[seat].n_value + 2.0)
                showdown_rate = (stats[seat].n_showdowns + 1.0) / (stats[seat].n_hands + 2.0)
                bettor_hand_aggr = per_seat_aggressive_this_hand[seat]

                features = np.array([
                    1.5 if action == ALL_IN else 1.0,
                    board_str,
                    prior_aggression,
                    1.0 if seat > 0 else 0.0,  # crude position proxy
                    pot_committed,
                    n_bets_this_street + 1,
                    1.0 if action == ALL_IN else 0.0,
                    0.0,
                    bluff_rate,
                    showdown_rate,
                    float(bettor_hand_aggr),
                    n_bets_this_street / max(1, decisions_this_street),
                ], dtype=float)

                Xs.append(features)
                ys.append(label)

            # Update per-hand counters
            per_seat_actions_so_far[seat] += 1
            per_seat_actions_this_hand[seat] += 1
            if action in (POT_BET, ALL_IN):
                per_seat_aggressive_so_far[seat] += 1
                per_seat_aggressive_this_hand[seat] += 1

        # Roll up per-hand stats to lifetime stats.
        for seat in range(n):
            stats[seat].n_actions += per_seat_actions_this_hand[seat]
            stats[seat].n_aggressive += per_seat_aggressive_this_hand[seat]
            stats[seat].n_hands += 1
            if seat in showdown_seats:
                stats[seat].n_showdowns += 1
                # Also count bluffs/value for future feature lookups
                if state.board:
                    label = _hand_strength_label(state.hole_cards[seat], state.board)
                    if label == 1:
                        stats[seat].n_bluffs += 1
                    elif label == 0:
                        stats[seat].n_value += 1

    if not Xs:
        return np.zeros((0, 12)), np.zeros((0,), dtype=int)
    return np.stack(Xs), np.array(ys, dtype=int)


def build_nlhe_bluff_dataset(
    n_hands: int = 3000,
    seed: int = 0,
    config: NLHEConfig | None = None,
) -> tuple:
    """Build a labeled bluff dataset from a 6-handed NLHE simulator."""
    from pokerbot.runtime import make_nlhe_player, NLHE_PROFILES

    config = config or NLHEConfig(n_players=6)
    rng = random.Random(seed)
    # Mix of profiles so the dataset isn't biased.
    names = list(NLHE_PROFILES.keys())
    players = []
    for i in range(config.n_players):
        name = names[i % len(names)]
        players.append(make_nlhe_player(name, rng=random.Random(rng.randint(0, 2**31))))

    game = NLHE(config)
    return extract_nlhe_bluff_examples(game, players, rng, n_hands)
