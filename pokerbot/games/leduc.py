"""Leduc Hold'em — the standard mid-size benchmark for poker solvers.

Rules (Southey et al. 2005):
- Deck: 6 cards = {J, Q, K} x 2 suits. Each player dealt 1 hole card.
- Two rounds. Round 1 (preflop): players act, no community card.
- Round 2 (flop): one community card revealed.
- Each round: 1st action is 'check' or 'bet'. After a bet, opponent can
  fold/call/raise. Cap of 2 raises per round.
- Antes: 1 chip each. Bet sizes: round 1 = 2, round 2 = 4 (standard variant).
- Showdown: a pair (hole matches public) beats a non-pair; otherwise high card.

State representation:
    cards:  tuple of cards dealt so far. Length 0/1 = pre-deal, 2 = both hole
            cards dealt, 3 = community card on the board.
    history: a string of action tokens, with '/' separating betting rounds.

Action tokens (in this implementation):
    'k' = check
    'c' = call
    'b' = bet
    'r' = raise
    'f' = fold

Round 2 starts at history "...c/" or "...kk/" (a "completed" round 1).

This is the standard Leduc rule set used in CFR papers; total info sets ~= 936
per player (1872 unique infosets), ~few thousand legal histories.
"""
from __future__ import annotations

from typing import Any, Hashable

# 3 ranks x 2 "suits" -> 6 cards; suits don't matter for hand value, just for
# distinguishing duplicates. Encode as integer 0..5 where rank = card // 2.
RANKS = (0, 1, 2)
RANK_NAMES = {0: "J", 1: "Q", 2: "K"}
DECK = tuple(range(6))  # 0,1 = J; 2,3 = Q; 4,5 = K


def card_rank(card: int) -> int:
    return card // 2


def card_name(card: int) -> str:
    return f"{RANK_NAMES[card_rank(card)]}{card % 2}"


# Bet sizes per round
BET_SIZE = {1: 2, 2: 4}
ANTE = 1
MAX_RAISES_PER_ROUND = 2  # so a round is at most: bet, raise, raise=cap, then call/fold


# A state is (cards, history). cards is a tuple of ints. history is a string.
LeducState = tuple[tuple[int, ...], str]


class LeducPoker:
    NUM_PLAYERS = 2

    def initial_state(self) -> LeducState:
        return ((), "")

    # --- chance ---

    def is_chance(self, state: LeducState) -> bool:
        cards, history = state
        # Deal hole cards if we have <2 cards.
        if len(cards) < 2:
            return True
        # Deal community card after round 1 ends. Round 1 ends when history
        # contains exactly one '/'.
        round1_done = "/" in history
        if round1_done and len(cards) == 2:
            # Need to check the round 1 actually ended in call (not fold).
            # If somebody folded in round 1, we're terminal, not chance.
            return not self._round1_ended_in_fold(history)
        return False

    def chance_outcomes(self, state: LeducState) -> list[tuple[Any, float]]:
        cards, _ = state
        used = set(cards)
        remaining = [c for c in DECK if c not in used]
        p = 1.0 / len(remaining)
        return [((c,), p) for c in remaining]

    # --- actions ---

    def current_player(self, state: LeducState) -> int:
        _, history = state
        return self._actor_for(history)

    def legal_actions(self, state: LeducState) -> list[str]:
        _, history = state
        round_idx = self._round_index(history)
        round_history = self._round_history(history)
        # Count bet/raise actions to enforce cap.
        bets_this_round = sum(1 for ch in round_history if ch in ("b", "r"))

        if round_history == "":
            # First action of the round: check or bet.
            return ["k", "b"]
        last = round_history[-1]
        if last in ("b", "r"):
            # Facing a bet/raise. Options: fold, call, raise (if not capped).
            actions = ["f", "c"]
            if bets_this_round < MAX_RAISES_PER_ROUND:
                actions.append("r")
            return actions
        if last == "k":
            # Opponent checked first. Options: check (=close) or bet.
            return ["k", "b"]
        raise ValueError(f"Unexpected round_history end: {round_history!r}")

    def apply(self, state: LeducState, action) -> LeducState:
        cards, history = state
        if isinstance(action, tuple):  # chance outcome
            new_cards = cards + action
            new_history = history
            if len(new_cards) == 3 and "/" not in history:
                # shouldn't happen — community card only deals after a round-ender
                pass
            return (new_cards, new_history)
        new_history = history + action
        # If the action closes round 1, append '/' to mark the round break.
        if self._round_index(history) == 1 and self._round_just_closed(new_history):
            new_history = new_history + "/"
        return (cards, new_history)

    # --- termination ---

    def is_terminal(self, state: LeducState) -> bool:
        cards, history = state
        if not history:
            return False
        # Fold ends the game immediately.
        if "f" in history:
            return True
        # Round 2 closure: the round-2 history string has just been closed.
        if "/" in history:
            r2 = self._round_history(history)
            if r2 and self._round_just_closed_str(r2):
                return True
        return False

    def utility(self, state: LeducState, player: int) -> float:
        cards, history = state
        # Folder's payoff = -(amount they put in beyond ante? we count the pot).
        # Cleaner: compute pot contributions per player, winner takes pot - own
        # contribution; loser pays own contribution.
        contrib = self._contributions(history)
        pot = sum(contrib)

        if "f" in history:
            # Whoever folded loses their contribution; other wins it.
            # Folder is the player who took the 'f' action.
            folder = self._folder(history)
            winner = 1 - folder
        else:
            # Showdown.
            hole0, hole1, board = cards[0], cards[1], cards[2]
            winner = self._showdown_winner(hole0, hole1, board)
            if winner == -1:
                # Tie — split pot.
                # Each gets pot/2; their net = pot/2 - contrib[i]
                net0 = pot / 2 - contrib[0]
                net1 = pot / 2 - contrib[1]
                return net0 if player == 0 else net1

        loser = 1 - winner
        net_winner = pot - contrib[winner]
        net_loser = -contrib[loser]
        if player == winner:
            return net_winner
        return net_loser

    # --- info sets ---

    def info_set_key(self, state: LeducState, player: int) -> Hashable:
        cards, history = state
        my_card = cards[player]
        board = cards[2] if len(cards) >= 3 else None
        my_rank = RANK_NAMES[card_rank(my_card)]
        board_rank = RANK_NAMES[card_rank(board)] if board is not None else "_"
        return f"{my_rank}|{board_rank}|{history}"

    # --- helpers ---

    def _round_index(self, history: str) -> int:
        """1 if we're in round 1, 2 if round 2."""
        return 1 + history.count("/")

    def _round_history(self, history: str) -> str:
        if "/" in history:
            return history.split("/")[-1]
        return history

    def _round1_ended_in_fold(self, history: str) -> bool:
        if "/" not in history:
            r1 = history
        else:
            r1 = history.split("/")[0]
        return "f" in r1

    def _round_just_closed_str(self, round_str: str) -> bool:
        """True iff this round-string represents a closed round (no further action)."""
        if not round_str:
            return False
        last = round_str[-1]
        # Round closes on: 'kk' (check-check), 'c' after bet/raise, 'f' (fold).
        if last == "f":
            return True
        if last == "c":
            return True
        if round_str == "kk":
            return True
        return False

    def _round_just_closed(self, history: str) -> bool:
        return self._round_just_closed_str(self._round_history(history))

    def _actor_for(self, history: str) -> int:
        """Whose turn is it after `history`? P0 always acts first in each round."""
        round_history = self._round_history(history)
        # Number of actions taken IN THIS ROUND determines parity.
        # P0 acts when count is even, P1 when odd.
        return len(round_history) % 2

    def _contributions(self, history: str) -> list[int]:
        """Total chips each player has put in (incl. antes), given history.

        Walks the history maintaining per-round running totals so that
        'call' and 'raise' know how much to match.
        """
        contrib = [ANTE, ANTE]
        round_idx = 1
        actor = 0
        # Per-round contributions above the ante baseline.
        round_in = [0, 0]
        for ch in history:
            if ch == "/":
                round_idx = 2
                actor = 0
                round_in = [0, 0]
                continue
            if ch == "k":
                pass
            elif ch == "b":
                amount = BET_SIZE[round_idx]
                contrib[actor] += amount
                round_in[actor] += amount
            elif ch == "r":
                # Call to match opp's round_in, then raise by bet_size.
                opp = 1 - actor
                to_call = round_in[opp] - round_in[actor]
                add = to_call + BET_SIZE[round_idx]
                contrib[actor] += add
                round_in[actor] += add
            elif ch == "c":
                opp = 1 - actor
                to_call = round_in[opp] - round_in[actor]
                contrib[actor] += to_call
                round_in[actor] += to_call
            elif ch == "f":
                pass
            actor = 1 - actor
        return contrib

    def _folder(self, history: str) -> int:
        """Return the player who folded. Assumes 'f' in history."""
        actor = 0
        for ch in history:
            if ch == "/":
                actor = 0
                continue
            if ch == "f":
                return actor
            actor = 1 - actor
        raise ValueError("No fold in history")

    def _showdown_winner(self, hole0: int, hole1: int, board: int) -> int:
        """Return 0, 1, or -1 for tie."""
        r0, r1, rb = card_rank(hole0), card_rank(hole1), card_rank(board)
        # Pair beats non-pair.
        pair0 = (r0 == rb)
        pair1 = (r1 == rb)
        if pair0 and not pair1:
            return 0
        if pair1 and not pair0:
            return 1
        if pair0 and pair1:
            # Both paired: they paired the SAME rank; tie.
            return -1
        # No one paired: compare hole ranks.
        if r0 > r1:
            return 0
        if r1 > r0:
            return 1
        return -1
