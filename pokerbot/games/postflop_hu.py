"""Heads-up postflop NLHE — single street, abstracted via 5 strength buckets.

Treats the bot's flop decision as a self-contained game:
    - Pot is pre-existing (3 BB from preflop SRP).
    - Both players have 97 BB stacks.
    - Each player's "type" is their 5-bucket hand strength on this flop.
    - Action: K (check), F (fold), B (pot-bet), C (call), R (raise pot).
    - At showdown, equity = bucket_i_vs_bucket_j Monte Carlo runout estimate.

Cap raises at 3 to keep the tree finite. The resulting game has under
~200 info sets and CFR converges fast.

Notes:
    - This is a SINGLE-STREET solve. We pretend that after the action closes
      on the flop, the hand goes straight to showdown (or the runout is
      auto-completed by the equity matrix). Real postflop has two more streets,
      but a single-street solver gives a workable approximation: the GTO bet/
      check/raise frequencies on the flop are the dominant strategic choice.
    - The bucket distribution is treated as uniform over 5 buckets per player
      at the start. Real flops produce non-uniform bucket distributions; this
      is the standard imperfect-recall abstraction.
"""
from __future__ import annotations

from typing import Hashable

from pokerbot.abstraction.postflop_buckets import NUM_POSTFLOP_BUCKETS


# Action tokens
FOLD = "f"
CHECK = "k"
CALL = "c"
BET = "b"
RAISE = "r"


# Postflop pot starts at 3 BB (preflop SRP).
_INITIAL_POT = 3.0
_BET_POT_FROM_3 = 3.0   # pot-bet on a 3-BB pot is 3 BB
_RAISE_TO_AFTER_BET = 9.0  # raise pot after a 3-BB bet → +9 BB committed
_REBET_TO_AFTER_RAISE = 27.0  # 3-bet


# Pre-existing investment per player at start of postflop (each put 1.5 BB pre).
_INITIAL_IN = 1.5

COMMITMENTS = {
    # history -> (player0_in, player1_in)
    "":     (_INITIAL_IN, _INITIAL_IN),
    "k":    (_INITIAL_IN, _INITIAL_IN),
    "kk":   (_INITIAL_IN, _INITIAL_IN),                    # showdown 3-BB pot
    "kb":   (_INITIAL_IN, _INITIAL_IN + _BET_POT_FROM_3),
    "kbf":  (_INITIAL_IN, _INITIAL_IN + _BET_POT_FROM_3),
    "kbc":  (_INITIAL_IN + _BET_POT_FROM_3, _INITIAL_IN + _BET_POT_FROM_3),
    "kbr":  (_INITIAL_IN + _RAISE_TO_AFTER_BET, _INITIAL_IN + _BET_POT_FROM_3),
    "kbrf": (_INITIAL_IN + _RAISE_TO_AFTER_BET, _INITIAL_IN + _BET_POT_FROM_3),
    "kbrc": (_INITIAL_IN + _RAISE_TO_AFTER_BET, _INITIAL_IN + _RAISE_TO_AFTER_BET),
    "kbrr": (_INITIAL_IN + _RAISE_TO_AFTER_BET,
             _INITIAL_IN + _REBET_TO_AFTER_RAISE),
    "kbrrf": (_INITIAL_IN + _RAISE_TO_AFTER_BET,
              _INITIAL_IN + _REBET_TO_AFTER_RAISE),
    "kbrrc": (_INITIAL_IN + _REBET_TO_AFTER_RAISE,
              _INITIAL_IN + _REBET_TO_AFTER_RAISE),
    "b":    (_INITIAL_IN + _BET_POT_FROM_3, _INITIAL_IN),
    "bf":   (_INITIAL_IN + _BET_POT_FROM_3, _INITIAL_IN),
    "bc":   (_INITIAL_IN + _BET_POT_FROM_3, _INITIAL_IN + _BET_POT_FROM_3),
    "br":   (_INITIAL_IN + _BET_POT_FROM_3, _INITIAL_IN + _RAISE_TO_AFTER_BET),
    "brf":  (_INITIAL_IN + _BET_POT_FROM_3, _INITIAL_IN + _RAISE_TO_AFTER_BET),
    "brc":  (_INITIAL_IN + _RAISE_TO_AFTER_BET, _INITIAL_IN + _RAISE_TO_AFTER_BET),
    "brr":  (_INITIAL_IN + _REBET_TO_AFTER_RAISE,
             _INITIAL_IN + _RAISE_TO_AFTER_BET),
    "brrf": (_INITIAL_IN + _REBET_TO_AFTER_RAISE,
             _INITIAL_IN + _RAISE_TO_AFTER_BET),
    "brrc": (_INITIAL_IN + _REBET_TO_AFTER_RAISE,
             _INITIAL_IN + _REBET_TO_AFTER_RAISE),
}


# Terminal sequences.
TERMINAL_SEQS = {
    "kk":    ("showdown", None),
    "kbf":   ("fold", 1),     # P0 folded → P1 wins
    "kbc":   ("showdown", None),
    "kbrf":  ("fold", 0),     # P1 folded
    "kbrc":  ("showdown", None),
    "kbrrf": ("fold", 1),     # P0 folded
    "kbrrc": ("showdown", None),
    "bf":    ("fold", 0),     # P1 folded
    "bc":    ("showdown", None),
    "brf":   ("fold", 1),     # P0 folded
    "brc":   ("showdown", None),
    "brrf":  ("fold", 0),     # P1 folded
    "brrc":  ("showdown", None),
}


def _n_raises(history: str) -> int:
    """Count of bets/raises in this history."""
    return sum(1 for ch in history if ch in (BET, RAISE))


PostflopState = tuple  # ((bucket0, bucket1), history)


class PostflopHUGame:
    """Heads-up single-street postflop with bucket abstraction."""

    NUM_PLAYERS = 2

    def __init__(self, bucket_equity_matrix: list):
        self.bucket_equity = bucket_equity_matrix

    # --- chance ---

    def initial_state(self) -> PostflopState:
        return ((), "")

    def is_chance(self, state: PostflopState) -> bool:
        buckets, _ = state
        return len(buckets) < 2

    def chance_outcomes(self, state: PostflopState) -> list:
        return [((b,), 1.0 / NUM_POSTFLOP_BUCKETS)
                for b in range(NUM_POSTFLOP_BUCKETS)]

    # --- decisions ---

    def current_player(self, state: PostflopState) -> int:
        _, history = state
        return len(history) % 2

    def legal_actions(self, state: PostflopState) -> list:
        _, history = state
        if history == "":
            return [CHECK, BET]
        last = history[-1]
        if last == CHECK:
            return [CHECK, BET]
        if last in (BET, RAISE):
            actions = [FOLD, CALL]
            if _n_raises(history) < 3:
                actions.append(RAISE)
            return actions
        return []

    def apply(self, state: PostflopState, action) -> PostflopState:
        buckets, history = state
        if isinstance(action, tuple):  # chance
            return (buckets + action, history)
        return (buckets, history + action)

    def is_terminal(self, state: PostflopState) -> bool:
        buckets, history = state
        if len(buckets) < 2:
            return False
        return history in TERMINAL_SEQS

    def utility(self, state: PostflopState, player: int) -> float:
        buckets, history = state
        info = TERMINAL_SEQS.get(history)
        if info is None:
            raise ValueError(f"utility() at non-terminal {history!r}")
        kind, fold_winner = info
        p0_in, p1_in = COMMITMENTS[history]
        b0, b1 = buckets

        if kind == "fold":
            if fold_winner == 0:
                payoff_p0 = p1_in - _INITIAL_IN
            else:
                payoff_p0 = -(p0_in - _INITIAL_IN)
            return payoff_p0 if player == 0 else -payoff_p0

        # Showdown: equity-weighted.
        pot = p0_in + p1_in
        eq0 = self.bucket_equity[b0][b1]
        net_p0 = eq0 * pot - p0_in
        net_p1 = (1 - eq0) * pot - p1_in
        return net_p0 if player == 0 else net_p1

    def info_set_key(self, state: PostflopState, player: int) -> Hashable:
        buckets, history = state
        my_bucket = buckets[player]
        return f"pf{my_bucket}|{history}"
