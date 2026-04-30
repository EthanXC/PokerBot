"""No-Limit Texas Hold'em — N-player engine with action abstraction.

Supports 2-10 players. The complete unabstracted game has ~10^160 states
(Pluribus-scale), so we abstract aggressively:

ACTION ABSTRACTION
------------------
At every decision the player picks from at most 5 abstract actions:
    - fold      ('f')
    - check     ('k')        when no outstanding bet
    - call      ('c')        match the current outstanding bet
    - bet/raise ('p')        a pot-sized bet (or pot-sized raise)
    - all-in    ('a')        commit the entire stack

Pot-sized raise definition (the Pluribus convention):
    raise_to = current_bet + pot_after_call
This makes the bet sized to the pot _after_ we call — the standard "pot raise."

NOTE we also accept a half-pot bet ('h') optionally; off by default to keep
the tree small.

CARD ABSTRACTION (separate, in pokerbot/abstraction/)
----------------------------------------------------
Hole cards bucketed by preflop equity vs random opponents. The state itself
keeps actual cards; the info_set_key uses buckets so CFR can share strategy
across equity-equivalent hands.

STATE
-----
NLHEState is a frozen dataclass with everything needed to:
    - resume the game (apply legal action -> new state)
    - render an info-set key for the current player
    - compute terminal payoffs (with side pots when stacks differ)

A hand starts: blinds posted, hole cards dealt, action begins UTG (3+ players)
or with the dealer (heads-up). Round 0 = preflop. After action closes, the
dealer either deals the next round's community cards or settles the hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

from pokerbot.core.cards import Card, Deck, VALID_RANKS, VALID_SUITS
from pokerbot.core.evaluator import HandEvaluator


# Action tokens
FOLD = "f"
CHECK = "k"
CALL = "c"
POT_BET = "p"
HALF_POT_BET = "h"
ALL_IN = "a"

ALL_ACTIONS = (FOLD, CHECK, CALL, POT_BET, HALF_POT_BET, ALL_IN)

# Default cash-game parameters
DEFAULT_SMALL_BLIND = 1
DEFAULT_BIG_BLIND = 2
DEFAULT_STACK = 200          # 100 BB
DEFAULT_INCLUDE_HALF_POT = False


@dataclass(frozen=True)
class NLHEConfig:
    n_players: int = 6
    small_blind: int = DEFAULT_SMALL_BLIND
    big_blind: int = DEFAULT_BIG_BLIND
    starting_stack: int = DEFAULT_STACK
    include_half_pot: bool = DEFAULT_INCLUDE_HALF_POT

    def __post_init__(self):
        if not 2 <= self.n_players <= 10:
            raise ValueError(f"n_players must be 2..10, got {self.n_players}")


@dataclass(frozen=True)
class NLHEState:
    """Frozen state for an NLHE hand.

    All fields are tuples/frozensets/strs so the state is hashable. Copying
    is cheap because Python tuples are immutable views; we only re-build
    the tuples that change in `apply()`.

    Important: `to_act_idx` is the seat index of the next decision-maker.
    A hand is terminal iff all but one player folded OR the river action
    closed. We model the "river action closed" check explicitly.
    """
    config: NLHEConfig
    button: int                                    # dealer button position
    hole_cards: tuple                              # tuple of (Card, Card) per player
    board: tuple                                   # cards dealt so far on board
    stacks: tuple                                  # remaining stack per player
    contributed: tuple                             # chips put in this hand per player
    contributed_this_round: tuple                  # chips this round per player
    folded: frozenset                              # set of seat indices folded
    all_in: frozenset                              # set of seat indices all-in
    round_idx: int                                 # 0=preflop 1=flop 2=turn 3=river
    bet_to_match: int                              # outstanding bet (this round)
    last_raise_size: int                           # size of last bet/raise (for min-raise)
    actor: int                                     # whose turn
    last_aggressor: int                            # seat that bet/raised most recently
    has_acted_this_round: frozenset                # seats that have acted in this round
    # action history per round, for info-set keying
    history: tuple                                 # tuple-of-tuples: history[r] = ((seat, action), ...)


def initial_deal(
    config: NLHEConfig,
    deck_order: list,
    button: int = 0,
) -> NLHEState:
    """Deal 2 hole cards to each player and post blinds."""
    if len(deck_order) < 2 * config.n_players + 5:
        raise ValueError("not enough cards")

    n = config.n_players
    holes = [None] * n
    for seat in range(n):
        holes[seat] = (deck_order[2 * seat], deck_order[2 * seat + 1])

    # Post blinds. SB is seat (button+1) % n, BB is (button+2) % n in 3+ player.
    # Heads-up special case: button posts the SB and acts FIRST preflop.
    if n == 2:
        sb_seat = button
        bb_seat = (button + 1) % n
    else:
        sb_seat = (button + 1) % n
        bb_seat = (button + 2) % n

    stacks = [config.starting_stack] * n
    contributed = [0] * n
    contributed_round = [0] * n

    sb = min(config.small_blind, stacks[sb_seat])
    bb = min(config.big_blind, stacks[bb_seat])
    stacks[sb_seat] -= sb
    stacks[bb_seat] -= bb
    contributed[sb_seat] = sb
    contributed[bb_seat] = bb
    contributed_round[sb_seat] = sb
    contributed_round[bb_seat] = bb

    # First to act preflop:
    #   2 players: button (the SB) acts first preflop
    #   3+ players: UTG = (button+3) % n
    if n == 2:
        first = sb_seat
    else:
        first = (button + 3) % n

    return NLHEState(
        config=config,
        button=button,
        hole_cards=tuple(holes),
        board=(),
        stacks=tuple(stacks),
        contributed=tuple(contributed),
        contributed_this_round=tuple(contributed_round),
        folded=frozenset(),
        all_in=frozenset(),
        round_idx=0,
        bet_to_match=bb,
        last_raise_size=bb,
        actor=first,
        last_aggressor=bb_seat,
        has_acted_this_round=frozenset(),
        history=((),),  # one empty tuple for the preflop round
    )


def _next_active_seat(state: NLHEState, start: int) -> int:
    """Return the next seat index after `start` that hasn't folded or gone all-in."""
    n = state.config.n_players
    seat = (start + 1) % n
    for _ in range(n):
        if seat not in state.folded and seat not in state.all_in:
            return seat
        seat = (seat + 1) % n
    return start  # nobody else can act; caller should handle


def _round_is_closed(state: NLHEState) -> bool:
    """A round closes when:
      - all non-folded, non-allin players have acted at least once this round, AND
      - they've all matched the current bet_to_match (or are all-in).
    """
    n = state.config.n_players
    eligible = [
        s for s in range(n)
        if s not in state.folded and s not in state.all_in
    ]
    if not eligible:
        return True
    # Have all eligible players acted at least once this round?
    for s in eligible:
        if s not in state.has_acted_this_round:
            return False
    # And do they all match bet_to_match?
    for s in eligible:
        if state.contributed_this_round[s] != state.bet_to_match:
            return False
    return True


def _hand_over_by_folds(state: NLHEState) -> bool:
    """Only one player remains who hasn't folded — they win uncontested."""
    n = state.config.n_players
    remaining = [s for s in range(n) if s not in state.folded]
    return len(remaining) <= 1


class NLHE:
    """N-player No-Limit Hold'em engine."""

    NUM_PLAYERS = 6  # default; override per-instance via config

    def __init__(self, config: NLHEConfig | None = None):
        self.config = config or NLHEConfig()
        # Override the class attribute on this instance.
        self.NUM_PLAYERS = self.config.n_players

    # --- public API ---

    def initial_state(self, deck_order: list = None, button: int = 0) -> NLHEState:
        if deck_order is None:
            # Default: shuffled standard deck
            import random
            d = Deck()
            cards = d.remaining()
            rng = random.Random()
            rng.shuffle(cards)
            deck_order = cards
        return initial_deal(self.config, deck_order, button=button)

    def is_terminal(self, state: NLHEState) -> bool:
        if _hand_over_by_folds(state):
            return True
        if state.round_idx >= 4:
            return True
        # River closed with full board → showdown.
        if (
            state.round_idx == 3
            and len(state.board) == 5
            and _round_is_closed(state)
        ):
            return True
        # Everyone left is all-in (no decisions remain) and the board is full.
        n = state.config.n_players
        deciders = [
            s for s in range(n)
            if s not in state.folded and s not in state.all_in
        ]
        if not deciders and len(state.board) == 5:
            return True
        return False

    def is_chance(self, state: NLHEState) -> bool:
        """Chance node iff we need more board cards.

        We deal cards eagerly when:
          - the current round is closed, OR
          - everyone left is all-in (run it out)
        """
        if self.is_terminal(state):
            return False
        n = state.config.n_players
        deciders = [
            s for s in range(n)
            if s not in state.folded and s not in state.all_in
        ]
        no_decisions = len(deciders) == 0

        if not _round_is_closed(state) and not no_decisions:
            return False

        target = (3, 4, 5, 5)[min(state.round_idx, 3)]
        return len(state.board) < target

    def chance_outcomes(self, state: NLHEState) -> list:
        """Deal the next required community card(s).

        We deal them ONE AT A TIME — each chance "outcome" is one card. This
        keeps the chance-branching factor sane (52 -> ~45 cards).
        """
        used = set()
        for hc in state.hole_cards:
            used.update(hc)
        used.update(state.board)
        deck = [Card(r, s) for s in VALID_SUITS for r in VALID_RANKS]
        remaining = [c for c in deck if c not in used]
        if not remaining:
            return []
        p = 1.0 / len(remaining)
        return [((c,), p) for c in remaining]

    def current_player(self, state: NLHEState) -> int:
        return state.actor

    def legal_actions(self, state: NLHEState) -> list:
        a = state.actor
        actions = []
        # Fold is always legal when there's a bet to call.
        # Check if no bet to match (or we already match).
        owed = state.bet_to_match - state.contributed_this_round[a]
        my_stack = state.stacks[a]

        if owed > 0:
            actions.append(FOLD)
            if my_stack >= owed:
                actions.append(CALL)
        else:
            actions.append(CHECK)

        # Bet/raise: legal if we have stack to do it AND it's at least min-raise.
        # Compute pot-sized bet target.
        pot_now = sum(state.contributed)
        # After we call, the pot is pot_now + owed. Pot-bet adds another (pot_now+owed)
        # on top of our call. Total chips in: owed + (pot_now + owed) = pot_now + 2*owed.
        pot_bet_total = owed + (pot_now + owed)  # chips we put in beyond what we already have
        if my_stack > owed and pot_bet_total <= my_stack:
            # And: must be at least min-raise (last_raise_size beyond bet_to_match)
            min_raise_total = owed + state.last_raise_size
            if pot_bet_total >= min_raise_total:
                actions.append(POT_BET)

        if self.config.include_half_pot:
            half_pot_total = owed + (pot_now + owed) // 2
            if my_stack > owed and half_pot_total <= my_stack and half_pot_total >= owed + state.last_raise_size:
                actions.append(HALF_POT_BET)

        # All-in: legal if we have any stack (and going all-in beats just calling
        # — i.e., it adds chips beyond the call).
        if my_stack > 0:
            # All-in is "different" from call only if my_stack > owed.
            if my_stack > owed:
                actions.append(ALL_IN)
            elif my_stack <= owed and FOLD in actions:
                # We can still go all-in to call short.
                actions.append(ALL_IN)
        return actions

    def apply(self, state: NLHEState, action) -> NLHEState:
        # Chance action: tuple containing one Card.
        if isinstance(action, tuple):
            new_board = state.board + action
            return state._replace_board(new_board)._maybe_advance_round()

        return self._apply_decision(state, action)

    # --- internal helpers ---

    def _apply_decision(self, state: NLHEState, action: str) -> NLHEState:
        a = state.actor
        owed = state.bet_to_match - state.contributed_this_round[a]
        my_stack = state.stacks[a]
        new_folded = state.folded
        new_all_in = state.all_in
        new_stacks = list(state.stacks)
        new_contrib = list(state.contributed)
        new_contrib_round = list(state.contributed_this_round)
        new_bet_to_match = state.bet_to_match
        new_last_raise = state.last_raise_size
        new_last_aggressor = state.last_aggressor

        if action == FOLD:
            new_folded = state.folded | {a}
        elif action == CHECK:
            if owed != 0:
                raise ValueError(f"check illegal at seat {a}: owed={owed}")
            # Pure no-op chip-wise.
        elif action == CALL:
            put_in = min(owed, my_stack)
            new_stacks[a] -= put_in
            new_contrib[a] += put_in
            new_contrib_round[a] += put_in
            if new_stacks[a] == 0:
                new_all_in = state.all_in | {a}
        elif action == POT_BET:
            pot_now = sum(state.contributed)
            chips_in = owed + (pot_now + owed)
            if chips_in > my_stack:
                # Snap to all-in
                chips_in = my_stack
            new_stacks[a] -= chips_in
            new_contrib[a] += chips_in
            new_contrib_round[a] += chips_in
            new_bet_to_match = new_contrib_round[a]
            new_last_raise = chips_in - owed
            new_last_aggressor = a
            if new_stacks[a] == 0:
                new_all_in = state.all_in | {a}
        elif action == HALF_POT_BET:
            pot_now = sum(state.contributed)
            chips_in = owed + (pot_now + owed) // 2
            if chips_in > my_stack:
                chips_in = my_stack
            new_stacks[a] -= chips_in
            new_contrib[a] += chips_in
            new_contrib_round[a] += chips_in
            new_bet_to_match = new_contrib_round[a]
            new_last_raise = chips_in - owed
            new_last_aggressor = a
            if new_stacks[a] == 0:
                new_all_in = state.all_in | {a}
        elif action == ALL_IN:
            chips_in = my_stack
            new_stacks[a] = 0
            new_contrib[a] += chips_in
            new_contrib_round[a] += chips_in
            new_all_in = state.all_in | {a}
            if new_contrib_round[a] > new_bet_to_match:
                # all-in is also a raise
                new_last_raise = new_contrib_round[a] - new_bet_to_match
                new_bet_to_match = new_contrib_round[a]
                new_last_aggressor = a
        else:
            raise ValueError(f"unknown action {action!r}")

        # Track that this seat acted this round.
        new_has_acted = state.has_acted_this_round | {a}

        # Append to history.
        new_history_round = state.history[state.round_idx] + ((a, action),)
        new_history = state.history[:state.round_idx] + (new_history_round,) + state.history[state.round_idx + 1:]

        # If a raise happened, reset has_acted for everyone except actor —
        # they need to respond again.
        if action in (POT_BET, HALF_POT_BET) or (action == ALL_IN and new_last_aggressor == a):
            new_has_acted = frozenset({a})

        # Decide next actor.
        next_state = NLHEState(
            config=state.config,
            button=state.button,
            hole_cards=state.hole_cards,
            board=state.board,
            stacks=tuple(new_stacks),
            contributed=tuple(new_contrib),
            contributed_this_round=tuple(new_contrib_round),
            folded=new_folded,
            all_in=new_all_in,
            round_idx=state.round_idx,
            bet_to_match=new_bet_to_match,
            last_raise_size=new_last_raise,
            actor=state.actor,  # placeholder; recomputed below
            last_aggressor=new_last_aggressor,
            has_acted_this_round=new_has_acted,
            history=new_history,
        )
        return next_state._advance_actor()

    def utility(self, state: NLHEState, player: int) -> float:
        """Return chips P0..Pn won/lost in this hand, net of contributions.

        Handles side pots when stacks differ.
        """
        if not self.is_terminal(state):
            raise ValueError("utility() called on non-terminal state")
        n = state.config.n_players
        contrib = list(state.contributed)
        folded = state.folded

        # If only one player remains (everyone else folded), they win the pot.
        remaining = [s for s in range(n) if s not in folded]
        if len(remaining) == 1:
            winner = remaining[0]
            pot = sum(contrib)
            net = [-contrib[s] for s in range(n)]
            net[winner] += pot
            return float(net[player])

        # Showdown with possible side pots.
        # Algorithm: for each unique contribution level among non-folded players,
        # form a side pot and award it to the best non-folded hand among those
        # who contributed at least that level.
        net = [-contrib[s] for s in range(n)]
        # Sort non-folded contribution levels.
        contestants = [s for s in range(n) if s not in folded]
        contrib_levels = sorted(set(contrib[s] for s in contestants))

        prev_level = 0
        # Also the folded players' contributions go to the FIRST side pot (no return).
        for level in contrib_levels:
            # Pot size at this level: each player who contributed at least `level`
            # contributes (level - prev_level) chips to it; folded players who
            # also contributed at least `level` give their (level - prev_level) chips.
            chip_per_player = level - prev_level
            pot_size = 0
            eligible_for_pot = []
            for s in range(n):
                # Anyone who put in at least `level`:
                contributed_to_this_pot = max(0, min(contrib[s], level) - prev_level)
                pot_size += contributed_to_this_pot
                # Eligible to win this pot only if NOT folded AND contributed at least `level`.
                if s not in folded and contrib[s] >= level:
                    eligible_for_pot.append(s)
            prev_level = level

            if pot_size <= 0 or not eligible_for_pot:
                continue

            # Find the best hand among eligible.
            scores = {s: HandEvaluator.score_seven(list(state.hole_cards[s]) + list(state.board))
                      for s in eligible_for_pot}
            best_score = max(scores.values())
            winners = [s for s, sc in scores.items() if sc == best_score]
            share = pot_size / len(winners)
            for w in winners:
                net[w] += share

        return float(net[player])

    def info_set_key(self, state: NLHEState, player: int) -> Hashable:
        """Hashable key identifying the player's view.

        Encoding: (hole-card-bucket, board-tuple, abstracted-history)
        """
        # Lazy: use raw cards by default (callers can override with abstraction)
        my_hole = state.hole_cards[player]
        # Abstract hole as a string for stability
        hole_str = "".join(sorted(str(c) for c in my_hole))
        board_str = "-".join(str(c) for c in state.board)
        history_str = "/".join(
            "".join(f"{seat}{act}" for seat, act in round_h)
            for round_h in state.history
        )
        return f"{hole_str}|{board_str}|{history_str}"


# --- Helper methods on NLHEState (added via monkey-patching to keep frozen) ---

def _replace_board(self: NLHEState, new_board: tuple) -> NLHEState:
    return NLHEState(
        config=self.config, button=self.button, hole_cards=self.hole_cards,
        board=tuple(new_board),
        stacks=self.stacks, contributed=self.contributed,
        contributed_this_round=self.contributed_this_round,
        folded=self.folded, all_in=self.all_in,
        round_idx=self.round_idx,
        bet_to_match=self.bet_to_match, last_raise_size=self.last_raise_size,
        actor=self.actor, last_aggressor=self.last_aggressor,
        has_acted_this_round=self.has_acted_this_round, history=self.history,
    )


def _advance_actor(self: NLHEState) -> NLHEState:
    """Find the next seat that needs to act. If round is closed, KEEP the
    current actor (the chance node will fire next via is_chance)."""
    if _hand_over_by_folds(self):
        return self
    if _round_is_closed(self):
        return self
    # Find next active seat (skipping folded + all-in).
    n = self.config.n_players
    s = (self.actor + 1) % n
    for _ in range(n):
        if s not in self.folded and s not in self.all_in:
            return NLHEState(
                config=self.config, button=self.button, hole_cards=self.hole_cards,
                board=self.board, stacks=self.stacks, contributed=self.contributed,
                contributed_this_round=self.contributed_this_round,
                folded=self.folded, all_in=self.all_in,
                round_idx=self.round_idx, bet_to_match=self.bet_to_match,
                last_raise_size=self.last_raise_size,
                actor=s, last_aggressor=self.last_aggressor,
                has_acted_this_round=self.has_acted_this_round, history=self.history,
            )
        s = (s + 1) % n
    return self


def _maybe_advance_round(self: NLHEState) -> NLHEState:
    """If we just completed a chance deal, check if we should advance round."""
    target_for_round = (0, 3, 4, 5)
    cur_round = self.round_idx
    # If we're past preflop and the board has just been brought up to the next
    # target, increment round_idx.
    if cur_round >= 4:
        return self
    expected = target_for_round[min(cur_round + 1, 3)]
    # If we're in round 0 and have 3 cards, advance to round 1.
    if cur_round == 0 and len(self.board) == 3:
        return self._start_new_round(1)
    if cur_round == 1 and len(self.board) == 4:
        return self._start_new_round(2)
    if cur_round == 2 and len(self.board) == 5:
        return self._start_new_round(3)
    if cur_round == 3 and len(self.board) == 5:
        # River played out; ready for showdown.
        return self._mark_terminal()
    return self


def _start_new_round(self: NLHEState, new_round: int) -> NLHEState:
    """Reset round-local fields and pick first to act (left of button)."""
    n = self.config.n_players
    # First to act post-flop: first seat after button who is still in the hand.
    seat = (self.button + 1) % n
    for _ in range(n):
        if seat not in self.folded and seat not in self.all_in:
            break
        seat = (seat + 1) % n
    new_history = self.history + ((),) if len(self.history) == new_round else self.history
    if len(new_history) <= new_round:
        new_history = new_history + ((),) * (new_round + 1 - len(new_history))
    return NLHEState(
        config=self.config, button=self.button, hole_cards=self.hole_cards,
        board=self.board,
        stacks=self.stacks,
        contributed=self.contributed,
        contributed_this_round=tuple([0] * n),
        folded=self.folded,
        all_in=self.all_in,
        round_idx=new_round,
        bet_to_match=0,
        last_raise_size=self.config.big_blind,
        actor=seat,
        last_aggressor=seat,
        has_acted_this_round=frozenset(),
        history=new_history,
    )


def _mark_terminal(self: NLHEState) -> NLHEState:
    """Advance round_idx to 4 to signal terminal."""
    return NLHEState(
        config=self.config, button=self.button, hole_cards=self.hole_cards,
        board=self.board, stacks=self.stacks, contributed=self.contributed,
        contributed_this_round=self.contributed_this_round,
        folded=self.folded, all_in=self.all_in,
        round_idx=4,
        bet_to_match=self.bet_to_match, last_raise_size=self.last_raise_size,
        actor=self.actor, last_aggressor=self.last_aggressor,
        has_acted_this_round=self.has_acted_this_round, history=self.history,
    )


# Attach helpers to NLHEState class.
NLHEState._replace_board = _replace_board
NLHEState._advance_actor = _advance_actor
NLHEState._maybe_advance_round = _maybe_advance_round
NLHEState._start_new_round = _start_new_round
NLHEState._mark_terminal = _mark_terminal
