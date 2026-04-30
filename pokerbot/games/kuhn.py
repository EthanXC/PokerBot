"""Kuhn poker — the smallest interesting poker game.

Rules:
- 3-card deck: J(1), Q(2), K(3). Higher card wins at showdown.
- Each player antes 1 chip and is dealt 1 card.
- Player 0 acts first. Actions: 'p' (pass/check) or 'b' (bet/call, +1 chip).
- The full action sequence (history) is one of:
    pp           -> both pass, showdown for the 2 antes
    pbp          -> P0 checks, P1 bets, P0 folds; P1 wins ante
    pbb          -> P0 checks, P1 bets, P0 calls; showdown for ante+1
    bp           -> P0 bets, P1 folds; P0 wins ante
    bb           -> P0 bets, P1 calls; showdown for ante+1

State representation: tuple (cards, history)
  - cards: tuple of length 0, 1, or 2 (filled by chance)
  - history: string of 'p' and 'b'

This state is hashable (tuple) and easy to print, which makes debugging
the solver much easier than a dataclass with __repr__.
"""
from __future__ import annotations

from typing import Any, Hashable

# Cards are integers 1..3 with 3 = K (best).
CARDS = (1, 2, 3)
CARD_NAMES = {1: "J", 2: "Q", 3: "K"}

# Actions
PASS = "p"
BET = "b"


# A state is just (cards, history). Both are tuples/strings -> hashable.
KuhnState = tuple[tuple[int, ...], str]


class KuhnPoker:
    NUM_PLAYERS = 2

    def initial_state(self) -> KuhnState:
        return ((), "")

    # --- chance ---

    def is_chance(self, state: KuhnState) -> bool:
        cards, _ = state
        return len(cards) < 2

    def chance_outcomes(self, state: KuhnState) -> list[tuple[Any, float]]:
        cards, _ = state
        if len(cards) == 0:
            return [((c,), 1 / 3) for c in CARDS]
        # 1 card already dealt -> deal one of the remaining two to player 1
        dealt = cards[0]
        remaining = [c for c in CARDS if c != dealt]
        return [((c,), 0.5) for c in remaining]

    # --- decisions ---

    def current_player(self, state: KuhnState) -> int:
        _, history = state
        # P0 acts at history "" and history "pb"; P1 acts at "p" and "b".
        return len(history) % 2

    def legal_actions(self, state: KuhnState) -> list[str]:
        return [PASS, BET]

    def apply(self, state: KuhnState, action) -> KuhnState:
        cards, history = state
        # Chance action is a 1-tuple (card,)
        if isinstance(action, tuple):
            return (cards + action, history)
        return (cards, history + action)

    # --- terminal logic ---

    def is_terminal(self, state: KuhnState) -> bool:
        cards, history = state
        if len(cards) < 2:
            return False
        return history in {"pp", "pbp", "pbb", "bp", "bb"}

    def utility(self, state: KuhnState, player: int) -> float:
        cards, history = state
        c0, c1 = cards
        # Showdown winner (only meaningful when terminal is a showdown).
        showdown_winner = 0 if c0 > c1 else 1

        if history == "pp":
            payoff_to_p0 = +1 if showdown_winner == 0 else -1  # 1 ante swap
        elif history == "pbp":
            payoff_to_p0 = -1   # P0 folded after P1 bet
        elif history == "pbb":
            payoff_to_p0 = +2 if showdown_winner == 0 else -2  # ante + bet
        elif history == "bp":
            payoff_to_p0 = +1   # P1 folded
        elif history == "bb":
            payoff_to_p0 = +2 if showdown_winner == 0 else -2
        else:
            raise ValueError(f"Not a terminal history: {history!r}")

        return payoff_to_p0 if player == 0 else -payoff_to_p0

    # --- info sets ---

    def info_set_key(self, state: KuhnState, player: int) -> Hashable:
        """A player's information set: their card + the public history."""
        cards, history = state
        my_card = cards[player]
        return f"{CARD_NAMES[my_card]}:{history}"
