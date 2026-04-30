"""Run a session of Leduc hands and record everything.

A HandTrace records every step of one hand:
  - the cards each player held (and the board card if dealt)
  - every (state, actor, action) tuple
  - per-player net chip change at the end

A session = a sequence of HandTrace objects with the running chip totals.

Why we need this: the bluff/tilt classifiers want labeled training examples,
and the labels come from things we can only know by knowing the actual cards
(was that bet a bluff? did that player just take a brutal beat?). Running
sessions ourselves means we *do* know everything.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from pokerbot.games.leduc import LeducPoker, card_rank
from pokerbot.runtime.heuristic_player import LeducHeuristicPlayer


@dataclass
class ActionRecord:
    """One decision in a hand."""
    actor: int                   # 0 or 1
    state_history: str           # game.history at decision time
    info_set: str                # info-set key from the actor's view
    action: str                  # action chosen
    legal_actions: tuple = ()    # (for feature extraction later)


@dataclass
class HandTrace:
    """Complete record of one hand."""
    hole_cards: tuple             # (p0_card, p1_card)
    board_card: int | None        # community card (or None if folded preflop)
    actions: list = field(default_factory=list)  # list[ActionRecord]
    p0_net: float = 0.0           # P0's chip delta
    p1_net: float = 0.0           # P1's chip delta
    went_to_showdown: bool = False
    winner: int | None = None     # 0/1/None for tie

    def actor_card(self, actor: int) -> int:
        return self.hole_cards[actor]


def _play_one_hand(
    game: LeducPoker,
    p0: LeducHeuristicPlayer,
    p1: LeducHeuristicPlayer,
    rng: random.Random,
) -> HandTrace:
    """Play exactly one hand of Leduc; return a HandTrace."""
    state = game.initial_state()
    actions: list[ActionRecord] = []

    # Walk through the hand: chance nodes auto-resolve, decision nodes ask players.
    while not game.is_terminal(state):
        if game.is_chance(state):
            outcomes = game.chance_outcomes(state)
            r = rng.random()
            cum = 0.0
            chosen = outcomes[-1][0]
            for action, p in outcomes:
                cum += p
                if r < cum:
                    chosen = action
                    break
            state = game.apply(state, chosen)
            continue

        actor = game.current_player(state)
        legal = game.legal_actions(state)
        info_set = game.info_set_key(state, actor)
        player = p0 if actor == 0 else p1
        chosen = player.decide(game, state, legal, actor)
        actions.append(
            ActionRecord(
                actor=actor,
                state_history=state[1],
                info_set=info_set,
                action=chosen,
                legal_actions=tuple(legal),
            )
        )
        state = game.apply(state, chosen)

    cards, history = state
    hole_cards = (cards[0], cards[1])
    board = cards[2] if len(cards) >= 3 else None

    p0_net = game.utility(state, 0)
    p1_net = game.utility(state, 1)

    # Was there a showdown? Iff no one folded.
    went_showdown = "f" not in history

    winner: int | None = None
    if went_showdown:
        # Re-derive winner via showdown logic.
        c0 = cards[0]; c1 = cards[1]; b = cards[2]
        # Mirror LeducPoker's showdown helper
        r0, r1, rb = card_rank(c0), card_rank(c1), card_rank(b)
        pair0 = (r0 == rb); pair1 = (r1 == rb)
        if pair0 and not pair1:
            winner = 0
        elif pair1 and not pair0:
            winner = 1
        elif pair0 and pair1:
            winner = None
        else:
            if r0 > r1:
                winner = 0
            elif r1 > r0:
                winner = 1
            else:
                winner = None
    else:
        # Folder loses; the other wins.
        # Find the folder.
        actor = 0
        for ch in history:
            if ch == "/":
                actor = 0; continue
            if ch == "f":
                winner = 1 - actor
                break
            actor = 1 - actor

    return HandTrace(
        hole_cards=hole_cards,
        board_card=board,
        actions=actions,
        p0_net=p0_net,
        p1_net=p1_net,
        went_to_showdown=went_showdown,
        winner=winner,
    )


def run_session(
    p0: LeducHeuristicPlayer,
    p1: LeducHeuristicPlayer,
    n_hands: int,
    rng: random.Random | None = None,
) -> list[HandTrace]:
    """Play n_hands of Leduc between p0 and p1. Return all traces."""
    rng = rng or random.Random()
    game = LeducPoker()
    traces: list[HandTrace] = []
    for _ in range(n_hands):
        trace = _play_one_hand(game, p0, p1, rng)
        # Feed each player the result so tilt mechanics can fire.
        p0.observe_result(trace.p0_net)
        p1.observe_result(trace.p1_net)
        traces.append(trace)
    return traces
