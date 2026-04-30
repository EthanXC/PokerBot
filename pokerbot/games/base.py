"""Abstract extensive-form game protocol.

CFR (and any other tree-search algorithm) only needs a handful of operations
on a game tree:

  - is_terminal(state)            -> bool
  - utility(state, player)        -> float           (only at terminals)
  - is_chance(state)              -> bool
  - chance_outcomes(state)        -> list[(action, prob)]
  - current_player(state)         -> int             (0 or 1 for two-player)
  - legal_actions(state)          -> list[Action]
  - apply(state, action)          -> new state
  - info_set_key(state, player)   -> hashable        (the player's view)

This module defines the protocol; concrete games (Kuhn, Leduc, NLHE)
implement it.

Why a protocol and not an ABC: keeps games as cheap value objects (often
just a tuple/string) so solver hot-paths don't pay attribute-access costs.
"""
from __future__ import annotations

from typing import Any, Generic, Hashable, Iterable, Protocol, TypeVar


# Generic state type — each game picks its own (string, tuple, dataclass, etc.)
State = TypeVar("State")
Action = Hashable


class GameState(Protocol):
    """Marker protocol — concrete game states should be hashable + immutable."""

    def __hash__(self) -> int: ...


class ExtensiveFormGame(Protocol[State]):
    """Two-player zero-sum extensive-form game with chance + imperfect info."""

    NUM_PLAYERS: int

    def initial_state(self) -> State: ...

    def is_terminal(self, state: State) -> bool: ...

    def utility(self, state: State, player: int) -> float:
        """Payoff to `player` at terminal `state`. Zero-sum: u0 + u1 = 0."""

    def is_chance(self, state: State) -> bool: ...

    def chance_outcomes(self, state: State) -> list[tuple[Any, float]]:
        """List of (action, probability) for chance nodes."""

    def current_player(self, state: State) -> int: ...

    def legal_actions(self, state: State) -> list[Action]: ...

    def apply(self, state: State, action: Action) -> State: ...

    def info_set_key(self, state: State, player: int) -> Hashable:
        """A hashable key identifying the info set from `player`'s perspective.

        Two states must map to the same key iff `player` cannot distinguish them.
        """


def all_info_sets(game: ExtensiveFormGame, player: int) -> dict:
    """Walk the entire tree and collect info-set keys + legal-action lists.

    Useful for small games (Kuhn, Leduc) to enumerate the strategy table.
    Returns a dict {info_set_key -> list[Action]}.
    """
    out: dict = {}
    _walk(game, game.initial_state(), player, out)
    return out


def _walk(game: ExtensiveFormGame, state, player: int, out: dict) -> None:
    if game.is_terminal(state):
        return
    if game.is_chance(state):
        for action, _p in game.chance_outcomes(state):
            _walk(game, game.apply(state, action), player, out)
        return
    actions = game.legal_actions(state)
    if game.current_player(state) == player:
        key = game.info_set_key(state, player)
        if key not in out:
            out[key] = list(actions)
    for a in actions:
        _walk(game, game.apply(state, a), player, out)
