"""Counterfactual Regret Minimization (Zinkevich 2007) + CFR+ (Tammelin 2014).

Vanilla CFR walks the entire game tree per iteration. That's fine for Kuhn
(30 leaves) and Leduc (~3,800 info sets); for NLHE we'll switch to MCCFR.

The two CFR+ tweaks are toggleable:
  - `plus_regret_floor=True`: clamp regrets at 0 each iteration
  - `linear_averaging=True`:   weight iter t's contribution to the average by t

When both are off, this is vanilla CFR exactly as in Zinkevich's paper.

API:
    solver = CFRSolver(KuhnPoker())
    solver.train(iterations=10_000)
    avg = solver.average_strategy()       # {info_set_key: {action: prob}}
"""
from __future__ import annotations

from typing import Hashable

from pokerbot.games.base import ExtensiveFormGame


def regret_match(regrets: list[float]) -> list[float]:
    """Standard regret-matching: probabilities proportional to positive regrets.

    If all regrets are <= 0, returns the uniform distribution.
    """
    pos = [max(0.0, r) for r in regrets]
    s = sum(pos)
    if s > 0:
        return [p / s for p in pos]
    n = len(regrets)
    return [1.0 / n] * n


class InfoSetTable:
    """Per-info-set storage of cumulative regrets and strategy sums."""

    __slots__ = ("regrets", "strategy_sum", "actions")

    def __init__(self, actions: list[Hashable]):
        self.actions = list(actions)
        self.regrets: list[float] = [0.0] * len(actions)
        self.strategy_sum: list[float] = [0.0] * len(actions)

    def current_strategy(self) -> list[float]:
        return regret_match(self.regrets)

    def average_strategy(self) -> list[float]:
        s = sum(self.strategy_sum)
        if s > 0:
            return [x / s for x in self.strategy_sum]
        n = len(self.actions)
        return [1.0 / n] * n


class CFRSolver:
    def __init__(
        self,
        game: ExtensiveFormGame,
        plus_regret_floor: bool = False,
        linear_averaging: bool = False,
    ):
        self.game = game
        self.tables: dict[Hashable, InfoSetTable] = {}
        self.plus_regret_floor = plus_regret_floor
        self.linear_averaging = linear_averaging
        self.iteration = 0

    # --- public ---

    def train(self, iterations: int, verbose_every: int | None = None) -> None:
        for t in range(1, iterations + 1):
            self.iteration = t
            for player in range(self.game.NUM_PLAYERS):
                self._cfr(self.game.initial_state(), player, [1.0, 1.0])
            if verbose_every and t % verbose_every == 0:
                print(f"[CFR] iter {t}/{iterations}, info sets: {len(self.tables)}")

    def average_strategy(self) -> dict[Hashable, dict[Hashable, float]]:
        out: dict[Hashable, dict[Hashable, float]] = {}
        for key, table in self.tables.items():
            probs = table.average_strategy()
            out[key] = {a: p for a, p in zip(table.actions, probs)}
        return out

    def current_strategy(self) -> dict[Hashable, dict[Hashable, float]]:
        out: dict[Hashable, dict[Hashable, float]] = {}
        for key, table in self.tables.items():
            probs = table.current_strategy()
            out[key] = {a: p for a, p in zip(table.actions, probs)}
        return out

    # --- internal ---

    def _get_or_create(self, key: Hashable, actions: list[Hashable]) -> InfoSetTable:
        t = self.tables.get(key)
        if t is None:
            t = InfoSetTable(actions)
            self.tables[key] = t
        return t

    def _cfr(self, state, training_player: int, reach: list[float]) -> float:
        """Return the counterfactual utility to `training_player` at `state`,
        and update regrets/strategy-sums in place.

        `reach` is [reach_p0, reach_p1] — the product over each player's actions
        of the probabilities they took to arrive here.
        """
        game = self.game

        if game.is_terminal(state):
            return game.utility(state, training_player)

        if game.is_chance(state):
            ev = 0.0
            for action, p in game.chance_outcomes(state):
                ev += p * self._cfr(game.apply(state, action), training_player, reach)
            return ev

        player = game.current_player(state)
        actions = game.legal_actions(state)
        key = game.info_set_key(state, player)
        table = self._get_or_create(key, actions)
        strategy = table.current_strategy()

        action_utils = [0.0] * len(actions)
        node_util = 0.0

        for i, action in enumerate(actions):
            new_reach = list(reach)
            new_reach[player] *= strategy[i]
            u = self._cfr(game.apply(state, action), training_player, new_reach)
            action_utils[i] = u
            node_util += strategy[i] * u

        # Regrets/strategy-sums only update for the training player.
        if player == training_player:
            # counterfactual reach prob = product of OTHER reaches (and chance,
            # which we treat as folded into the recursion's `ev` weighting).
            other = 1.0
            for p in range(game.NUM_PLAYERS):
                if p != training_player:
                    other *= reach[p]

            for i in range(len(actions)):
                regret = action_utils[i] - node_util
                table.regrets[i] += other * regret
                if self.plus_regret_floor and table.regrets[i] < 0:
                    table.regrets[i] = 0.0

            # Strategy sum weighted by training player's own reach.
            weight = reach[training_player]
            if self.linear_averaging:
                weight *= self.iteration
            for i in range(len(actions)):
                table.strategy_sum[i] += weight * strategy[i]

        return node_util
