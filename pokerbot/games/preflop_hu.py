"""Heads-up preflop NLHE — abstracted via 10 hand buckets.

Small, fully-CFR-solvable game (~70 info sets). Used to produce the bot's
preflop GTO baseline.

Rules (heads-up):
    - SB posts 0.5 BB; BB posts 1 BB. Both have 100 BB stacks (effectively
      cap=100 BB but our abstraction caps re-raises before that).
    - SB acts first. Actions: F (fold), C (call to current bet), R (raise pot).
    - We cap at 3 total raises in the sequence to keep the tree finite.

Terminal payoffs:
    - F at any point: the folder loses what they have in the pot.
    - Showdown: pot is split per bucket-vs-bucket equity.

State:
    ((b0, b1), history) where b0,b1 are buckets (0..9) and history is
    the action string.

Sizes (in BB, treating BB=1 unit):
    Initial pot after blinds = 1.5
    Pot raise at the start: SB completes 0.5 then raises pot-after-call (= 2)
        => SB total in: 0.5 + 0.5 + 2 = 3
    Re-raise (3-bet): BB has 1 in, calls 2 more = 3, then raises pot-after-call
        (= 6) => BB total in: 9
    4-bet: SB has 3 in, calls 6 more = 9, then raises pot-after-call (= 18)
        => SB total in: 27

We use these specific commitment levels so each action's chip impact is fully
determined by its position in the sequence.
"""
from __future__ import annotations

from typing import Hashable

from pokerbot.abstraction.preflop_buckets import NUM_BUCKETS


# Action tokens
FOLD = "f"
CALL = "c"
RAISE = "r"


# Per-position commitment ladder (in BB units).
# After action sequence X, this is each player's TOTAL chips in:
#   "" -> SB=0.5, BB=1.0
#   "c" -> SB=1.0, BB=1.0
#   "ck" — wait, in this abstraction we don't have a separate "check" — when
#   facing no bet (after a limp), CALL means check. We're using one symbol.
# To keep clean: SB acting on "" can: F, C (limp = bring SB to 1), R (raise to 3).
# BB acting on "c" can: C (close round, showdown), R (raise to 3).
# BB acting on "r" can: F, C (call to 3), R (re-raise to 9).
# SB acting on "cr" can: F, C (call to 3), R (re-raise to 9).
# SB acting on "rr" can: F, C (call to 9), R (4-bet to 27).
# BB acting on "rrr" can: F, C (call to 27).  Cap reached.
# BB acting on "crrr" can: F, C (call to 27).  Cap reached.
#   Actually "crr" is 3 raises (limp+raise+reraise=2 raises... let me recount)
#   "c" = SB call, "cr" = SB call + BB raise (1 raise). "crr" = + SB reraise (2 raises).
#   "crrr" = + BB 4-bet (3 raises). Cap.

COMMITMENTS = {
    # history -> (sb_in, bb_in)
    "":      (0.5, 1.0),
    "f":     (0.5, 1.0),   # SB folded, BB wins 0.5
    "c":     (1.0, 1.0),
    "cc":    (1.0, 1.0),   # both checked, showdown for 2 BB
    "cr":    (1.0, 3.0),
    "crf":   (1.0, 3.0),   # SB folded after limp+raise
    "crc":   (3.0, 3.0),   # showdown
    "crr":   (9.0, 3.0),   # SB 3-bet
    "crrf":  (9.0, 3.0),   # BB folded
    "crrc":  (9.0, 9.0),   # showdown
    "crrr":  (9.0, 27.0),  # BB 4-bet (cap)
    "crrrf": (9.0, 27.0),  # SB folded
    "crrrc": (27.0, 27.0), # showdown
    "r":     (3.0, 1.0),
    "rf":    (3.0, 1.0),   # BB folded
    "rc":    (3.0, 3.0),   # showdown
    "rr":    (3.0, 9.0),
    "rrf":   (3.0, 9.0),
    "rrc":   (9.0, 9.0),   # showdown
    "rrr":   (27.0, 9.0),
    "rrrf":  (27.0, 9.0),
    "rrrc":  (27.0, 27.0),
}


# Terminal sequences and whether they are showdowns.
TERMINAL_SEQS = {
    "f":     ("fold", 1),     # SB folded, BB wins
    "rf":    ("fold", 0),     # BB folded, SB wins
    "rc":    ("showdown", None),
    "rr":    None,             # not terminal
    "rrf":   ("fold", 1),     # SB folded after BB 3-bet
    "rrc":   ("showdown", None),
    "rrrf":  ("fold", 0),     # BB folded after SB 4-bet
    "rrrc":  ("showdown", None),
    "c":     None,             # not terminal
    "cc":    ("showdown", None),  # SB called, BB checked
    "cr":    None,
    "crf":   ("fold", 1),     # SB folded after limp+raise
    "crc":   ("showdown", None),
    "crr":   None,
    "crrf":  ("fold", 0),     # BB folded
    "crrc":  ("showdown", None),
    "crrr":  None,
    "crrrf": ("fold", 1),
    "crrrc": ("showdown", None),
}


# Number of raises in the sequence so we can detect the cap.
def _n_raises(history: str) -> int:
    return history.count("r")


PreflopState = tuple  # ((bucket0, bucket1), history)


class PreflopHUGame:
    """Heads-up preflop with bucket abstraction. SB = player 0, BB = player 1."""

    NUM_PLAYERS = 2

    def __init__(self, bucket_equity_matrix: list):
        self.bucket_equity = bucket_equity_matrix

    # --- chance ---

    def initial_state(self) -> PreflopState:
        return ((), "")

    def is_chance(self, state: PreflopState) -> bool:
        buckets, _ = state
        return len(buckets) < 2

    def chance_outcomes(self, state: PreflopState) -> list:
        buckets, _ = state
        # Buckets are dealt independently uniform over NUM_BUCKETS each.
        # (Approximation: real preflop is correlated since cards differ.)
        return [((b,), 1.0 / NUM_BUCKETS) for b in range(NUM_BUCKETS)]

    # --- decisions ---

    def current_player(self, state: PreflopState) -> int:
        _, history = state
        # SB acts first; alternation by parity.
        return len(history) % 2

    def legal_actions(self, state: PreflopState) -> list:
        _, history = state
        actions = []
        # The first action ever (history "") cannot fold (SB can fold but it's
        # also legal preflop). Standard NLHE: SB CAN fold preflop.
        if history == "":
            return [FOLD, CALL, RAISE]
        last = history[-1]
        if last == "c":
            # No outstanding bet; CALL = check, RAISE legal.
            return [CALL, RAISE]
        if last == "r":
            # Outstanding raise. F / C / R unless cap reached.
            actions = [FOLD, CALL]
            if _n_raises(history) < 3:
                actions.append(RAISE)
            return actions
        return actions

    def apply(self, state: PreflopState, action) -> PreflopState:
        buckets, history = state
        if isinstance(action, tuple):  # chance
            return (buckets + action, history)
        return (buckets, history + action)

    def is_terminal(self, state: PreflopState) -> bool:
        buckets, history = state
        if len(buckets) < 2:
            return False
        info = TERMINAL_SEQS.get(history)
        return info is not None

    def utility(self, state: PreflopState, player: int) -> float:
        buckets, history = state
        info = TERMINAL_SEQS.get(history)
        if info is None:
            raise ValueError(f"utility() called on non-terminal {history!r}")
        sb_in, bb_in = COMMITMENTS[history]
        kind, fold_winner = info
        b0, b1 = buckets
        if kind == "fold":
            # fold_winner: 0 = SB wins, 1 = BB wins
            if fold_winner == 0:
                # SB wins. SB nets +bb_in, BB nets -bb_in.
                payoff_p0 = bb_in
            else:
                payoff_p0 = -sb_in
            return payoff_p0 if player == 0 else -payoff_p0
        # showdown: equity-weighted split. Each player nets equity*pot - their_in.
        pot = sb_in + bb_in
        eq_sb = self.bucket_equity[b0][b1]
        net_sb = eq_sb * pot - sb_in
        net_bb = (1 - eq_sb) * pot - bb_in
        return net_sb if player == 0 else net_bb

    def info_set_key(self, state: PreflopState, player: int) -> Hashable:
        buckets, history = state
        my_bucket = buckets[player]
        return f"b{my_bucket}|{history}"
