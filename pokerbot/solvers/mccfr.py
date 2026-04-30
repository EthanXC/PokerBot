"""Monte Carlo CFR — external sampling variant (Lanctot et al. 2009).

External sampling:
  - At chance nodes: sample one outcome (instead of summing over all).
  - At opponent nodes: sample one action (instead of summing).
  - At training-player nodes: enumerate ALL actions (so all regrets get updated).

This makes each iteration O(decision_depth * actions_per_node) instead of
O(tree_size). Variance is higher per iteration but it scales to games where
vanilla CFR is intractable.

Convergence: average strategy still converges to Nash; rate is similar in
practice to vanilla CFR after equating per-info-set update count.
"""
from __future__ import annotations

import random
from typing import Hashable

from pokerbot.games.base import ExtensiveFormGame
from pokerbot.solvers.cfr import InfoSetTable, regret_match


class MCCFRSolver:
    def __init__(
        self,
        game: ExtensiveFormGame,
        plus_regret_floor: bool = True,
        linear_averaging: bool = True,
        rng: random.Random | None = None,
    ):
        self.game = game
        self.tables: dict[Hashable, InfoSetTable] = {}
        self.plus_regret_floor = plus_regret_floor
        self.linear_averaging = linear_averaging
        self.rng = rng or random.Random()
        self.iteration = 0

    def train(self, iterations: int, verbose_every: int | None = None) -> None:
        for t in range(1, iterations + 1):
            self.iteration = t
            for player in range(self.game.NUM_PLAYERS):
                self._traverse(self.game.initial_state(), player)
            if verbose_every and t % verbose_every == 0:
                print(f"[MCCFR] iter {t}/{iterations}, info sets: {len(self.tables)}")

    def average_strategy(self) -> dict[Hashable, dict[Hashable, float]]:
        out: dict[Hashable, dict[Hashable, float]] = {}
        for key, table in self.tables.items():
            probs = table.average_strategy()
            out[key] = {a: p for a, p in zip(table.actions, probs)}
        return out

    # ---

    def _get_or_create(self, key: Hashable, actions: list[Hashable]) -> InfoSetTable:
        t = self.tables.get(key)
        if t is None:
            t = InfoSetTable(actions)
            self.tables[key] = t
        return t

    def _sample(self, choices_with_probs: list[tuple]) -> object:
        """Sample one (item, prob) by its probability."""
        r = self.rng.random()
        c = 0.0
        for item, p in choices_with_probs:
            c += p
            if r < c:
                return item
        return choices_with_probs[-1][0]  # numerical safety

    def _traverse(self, state, training_player: int) -> float:
        """Return EV to training_player for the sampled subtree, updating regrets."""
        game = self.game

        if game.is_terminal(state):
            return game.utility(state, training_player)

        if game.is_chance(state):
            outcomes = game.chance_outcomes(state)
            action = self._sample(outcomes)
            return self._traverse(game.apply(state, action), training_player)

        player = game.current_player(state)
        actions = game.legal_actions(state)
        key = game.info_set_key(state, player)
        table = self._get_or_create(key, actions)
        strategy = table.current_strategy()

        if player == training_player:
            # Enumerate all actions (this is what gives us full regret coverage).
            action_utils = [0.0] * len(actions)
            node_util = 0.0
            for i, a in enumerate(actions):
                u = self._traverse(game.apply(state, a), training_player)
                action_utils[i] = u
                node_util += strategy[i] * u

            # Update regrets — no opp/chance reach weighting (already implicit
            # in the sampling).
            for i in range(len(actions)):
                regret = action_utils[i] - node_util
                table.regrets[i] += regret
                if self.plus_regret_floor and table.regrets[i] < 0:
                    table.regrets[i] = 0.0

            # Strategy sum for the training player (linear averaging optional).
            weight = self.iteration if self.linear_averaging else 1.0
            for i in range(len(actions)):
                table.strategy_sum[i] += weight * strategy[i]

            return node_util
        else:
            # Opponent: sample one action.
            action = self._sample([(a, strategy[i]) for i, a in enumerate(actions)])
            # Even though we sample, we still want the opponent's strategy_sum
            # to track its own play (so we have an averaged opponent strategy too).
            # Following standard MCCFR-ES: we ALSO update opponent's strategy_sum.
            weight = self.iteration if self.linear_averaging else 1.0
            for i in range(len(actions)):
                table.strategy_sum[i] += weight * strategy[i]
            return self._traverse(game.apply(state, action), training_player)
