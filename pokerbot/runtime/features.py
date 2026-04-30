"""Extract labeled training examples from session traces.

Bluff examples
--------------
For every bet/raise that goes to showdown (so we see the bettor's hand):
    - Compute the 12 BluffClassifier features from context.
    - Label = 1 if the bet was a bluff, 0 if value.
      Bluff = the bettor did NOT have the strongest possible hand category
              for the spot (no pair preflop = bluff; no pair when board pairs
              = bluff; etc.).

Features 9-12 are temporal/opponent-context (added later as part of the
"richer features" improvement):
    9.  bettor_recent_bluff_rate    — empirical fraction of recent bets that
                                       were bluffs (computed from prior hands)
    10. bettor_showdown_rate        — fraction of recent hands that reached
                                       showdown (loose proxy for VPIP)
    11. hand_aggression_so_far      — # of bets/raises BY THE BETTOR earlier
                                       in this same hand (escalation detector)
    12. street_aggression_pace      — n_bets_this_street / decisions_this_street

Tilt examples
-------------
For every hand and each player, compute TiltClassifier features from
their pre-hand state, label = whether they're in a tilted condition
(big losses recently). Ground truth comes from the fact that we know
the player's recent results.

Mapping our 8 BluffClassifier features to Leduc:
    bet_size_ratio    : 1 if 'b', 1.5 if 'r' (raise = bigger commitment)
    board_strength    : 0.7 if board is K, 0.5 if Q, 0.3 if J, 0.0 if no board
    prior_aggression  : empirical recent agg freq for this player
    position          : 1.0 if bettor is the in-position player (P1) on this round
    pot_committed     : chips committed by bettor / starting stack proxy (clamped 0..1)
    n_bets_this_street: count of 'b'/'r' actions on this street so far
    overbet           : 1.0 if 'r' (more than min bet), else 0.0
    donk_lead         : 1.0 if bettor donk-led (P0 leads on round 2 here)

For tilt features we need a running window of stats per player. The
extractor walks traces in order to compute "recent" quantities relative
to each hand.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from pokerbot.games.leduc import card_rank
from pokerbot.runtime.session import HandTrace, ActionRecord


# -- Bluff feature extraction --

def _street_index(history: str) -> int:
    return 1 + history.count("/")


def _round_history(history: str) -> str:
    return history.split("/")[-1] if "/" in history else history


def _board_strength_from_card(board_card: int | None) -> float:
    if board_card is None:
        return 0.0
    r = card_rank(board_card)  # 0=J, 1=Q, 2=K
    return [0.3, 0.5, 0.7][r]


def _is_value_or_bluff_for_bettor(
    bettor_card: int, board_card: int | None
) -> int:
    """Return 1 if this is a 'bluff' (not the strongest hand category), 0 if value.

    Round 1 (no board): only K is a clean value bet; J/Q are bluffs/marginal.
    Round 2 (board dealt): a pair is value; anything else is bluff.
    """
    if board_card is None:
        return 0 if card_rank(bettor_card) == 2 else 1
    return 0 if card_rank(bettor_card) == card_rank(board_card) else 1


def extract_bluff_examples(traces: list[HandTrace]) -> tuple:
    """Return (X, y) numpy arrays for BluffClassifier training.

    Only includes bet/raise actions where the hand reached showdown
    (so we have ground truth for bluff vs. value).

    Returns 12 features per example (the original 8 plus 4 temporal/context
    ones — see module docstring).
    """
    Xs = []
    ys = []
    # Per-player rolling agg stats.
    agg_counts = [0, 0]  # bet/raise actions
    decision_counts = [0, 0]
    # Per-player rolling bluff/value counts (for bettor_recent_bluff_rate).
    bluff_counts = [0, 0]
    value_counts = [0, 0]
    # Per-player count of hands that reached showdown vs total.
    showdown_count = [0, 0]
    hands_count = [0, 0]

    for trace in traces:
        # Update hand-level counters first.
        for player in (0, 1):
            hands_count[player] += 1
            if trace.went_to_showdown:
                showdown_count[player] += 1

        if not trace.went_to_showdown:
            for a in trace.actions:
                decision_counts[a.actor] += 1
                if a.action in ("b", "r"):
                    agg_counts[a.actor] += 1
            continue

        # Per-bet feature extraction with per-actor "actions earlier in this hand"
        # tracking for the new escalation feature.
        n_bets_this_street = 0
        decisions_this_street = 0
        prev_street = 1
        bettor_actions_this_hand = [0, 0]  # b/r count by each actor in this hand

        for a in trace.actions:
            street = _street_index(a.state_history)
            if street != prev_street:
                n_bets_this_street = 0
                decisions_this_street = 0
                prev_street = street

            decision_counts[a.actor] += 1
            decisions_this_street += 1

            if a.action in ("b", "r"):
                bet_size_ratio = 1.5 if a.action == "r" else 1.0
                board_card = trace.board_card if street == 2 else None
                board_strength = _board_strength_from_card(board_card)
                prior_aggression = (
                    (agg_counts[a.actor] + 2.0)
                    / (decision_counts[a.actor] + 7.0)
                )
                position = 1.0 if a.actor == 1 else 0.0
                pot_committed = min(1.0, n_bets_this_street * 0.15 + 0.1)
                overbet = 1.0 if a.action == "r" else 0.0
                donk_lead = 1.0 if (street == 2 and a.actor == 0
                                    and _round_history(a.state_history) == "") else 0.0

                # ---- 4 new temporal/context features ----
                # 9. Bettor's empirical bluff rate (Beta(1,1)-shrunk so it's
                #    valid even with no observed bluffs/values yet).
                total_priors = bluff_counts[a.actor] + value_counts[a.actor]
                bettor_recent_bluff_rate = (
                    (bluff_counts[a.actor] + 1.0) / (total_priors + 2.0)
                )
                # 10. Bettor's showdown rate.
                bettor_showdown_rate = (
                    (showdown_count[a.actor] + 1.0) / (hands_count[a.actor] + 2.0)
                )
                # 11. How many bets/raises has the bettor put in earlier this hand?
                hand_aggression_so_far = float(bettor_actions_this_hand[a.actor])
                # 12. Pace of betting on this street.
                street_aggression_pace = (
                    n_bets_this_street / max(1, decisions_this_street)
                )

                features = np.array([
                    bet_size_ratio,
                    board_strength,
                    prior_aggression,
                    position,
                    pot_committed,
                    n_bets_this_street + 1,
                    overbet,
                    donk_lead,
                    bettor_recent_bluff_rate,
                    bettor_showdown_rate,
                    hand_aggression_so_far,
                    street_aggression_pace,
                ], dtype=float)

                board_card_at_bet = trace.board_card if street == 2 else None
                label = _is_value_or_bluff_for_bettor(
                    trace.actor_card(a.actor), board_card_at_bet
                )
                Xs.append(features)
                ys.append(label)

                # ---- Update rolling counters AFTER recording ----
                agg_counts[a.actor] += 1
                n_bets_this_street += 1
                bettor_actions_this_hand[a.actor] += 1
                if label == 1:
                    bluff_counts[a.actor] += 1
                else:
                    value_counts[a.actor] += 1

    if not Xs:
        return np.zeros((0, 12)), np.zeros((0,), dtype=int)
    return np.stack(Xs), np.array(ys, dtype=int)


# -- Tilt feature extraction --

@dataclass
class _PlayerWindow:
    """Per-player running stats used to build tilt features."""
    recent_results: deque
    recent_voluntary: deque
    recent_aggression: deque
    lifetime_voluntary: int = 0
    lifetime_aggressions: int = 0
    lifetime_actions: int = 0
    lifetime_3bets: int = 0
    lifetime_3bet_opportunities: int = 0
    hands_played: int = 0
    hands_since_loss: int = 100  # large = haven't lost recently


def _make_window() -> _PlayerWindow:
    return _PlayerWindow(
        recent_results=deque(maxlen=10),
        recent_voluntary=deque(maxlen=20),
        recent_aggression=deque(maxlen=20),
    )


def extract_tilt_examples(traces: list[HandTrace]) -> tuple:
    """Return (X, y) for TiltClassifier.

    For each player and each hand boundary, we build a 7-feature vector
    measuring "is this player on tilt right now?" The label comes from
    a synthetic ground-truth definition consistent with our heuristic
    player's tilt mechanic: y=1 iff cumulative chip swing in the last
    3 hands is below -5.
    """
    Xs = []
    ys = []
    windows = [_make_window(), _make_window()]

    for trace in traces:
        # Compute features BEFORE this hand for both players (their state on
        # entering the hand), then update the windows AFTER the hand resolves.
        for player in (0, 1):
            w = windows[player]
            if w.hands_played < 5:
                # Not enough history to be a useful example.
                continue

            recent_loss = -min(0.0, sum(w.recent_results))
            normalized_loss = min(1.0, recent_loss / 10.0)

            lifetime_vpip = w.lifetime_voluntary / max(1, w.hands_played)
            recent_vpip = (
                sum(w.recent_voluntary) / len(w.recent_voluntary)
                if w.recent_voluntary else lifetime_vpip
            )
            vpip_jump = recent_vpip - lifetime_vpip

            lifetime_agg = w.lifetime_aggressions / max(1, w.lifetime_actions)
            recent_agg = (
                sum(w.recent_aggression) / max(1, len(w.recent_aggression))
                if w.recent_aggression else lifetime_agg
            )
            agg_jump = recent_agg - lifetime_agg

            hands_since_loss = float(w.hands_since_loss)

            three_bet_jump = 0.0  # not tracked yet in Leduc, kept zero

            loss_streak = 0
            for r in reversed(w.recent_results):
                if r < 0: loss_streak += 1
                else: break

            vol_increase = 0.0  # placeholder — would compute std change

            features = np.array([
                normalized_loss,
                vpip_jump,
                agg_jump,
                hands_since_loss,
                three_bet_jump,
                float(loss_streak),
                vol_increase,
            ], dtype=float)

            # Ground truth label: same definition the heuristic player uses
            # internally — losing >5 chips in last 3 hands.
            if len(w.recent_results) >= 3:
                last3 = list(w.recent_results)[-3:]
                label = 1 if sum(last3) < -5.0 else 0
            else:
                label = 0
            Xs.append(features)
            ys.append(label)

        # --- Update windows AFTER capturing pre-hand features ---
        for player in (0, 1):
            w = windows[player]
            net = trace.p0_net if player == 0 else trace.p1_net
            w.recent_results.append(net)
            w.hands_played += 1
            if net < 0:
                w.hands_since_loss = 0
            else:
                w.hands_since_loss += 1

            # Did the player put money in voluntarily / take an aggressive action?
            was_voluntary = False
            actions_taken = 0
            aggressions = 0
            for a in trace.actions:
                if a.actor != player:
                    continue
                actions_taken += 1
                if a.action in ("b", "r"):
                    aggressions += 1
                if a.action in ("b", "c", "r"):
                    was_voluntary = True
            w.recent_voluntary.append(1 if was_voluntary else 0)
            w.recent_aggression.append(aggressions / max(1, actions_taken))
            if was_voluntary:
                w.lifetime_voluntary += 1
            w.lifetime_aggressions += aggressions
            w.lifetime_actions += actions_taken

    if not Xs:
        return np.zeros((0, 7)), np.zeros((0,), dtype=int)
    return np.stack(Xs), np.array(ys, dtype=int)
