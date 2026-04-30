"""HybridPolicy — the heart of the bot.

Combines a precomputed GTO blueprint with an exploitative best response
against a model of the opponent. Mix weight scales with how confident
we are in the opponent model.

    action_distribution(I) = (1 - lambda) * GTO(I) + lambda * BR_vs_model(I)

`lambda` should grow with data and shrink against unknowns. Specifically:

    lambda = base * confidence(opponent) * (1 - tilt_safety_factor)

For now, `confidence` = how much data we've seen (sigmoid-like). The
`tilt_safety_factor` reduces lambda when the opponent's behavior is
non-stationary (we don't trust the model's stability).

Key properties we want to verify with tests:
  1. lambda=0 → behavior identical to pure GTO.
  2. lambda=1 → behavior identical to pure best-response.
  3. Mixed strategy is a valid probability distribution at every info set.
  4. Hybrid never loses more EV vs. a balanced opponent than pure GTO does
     (within statistical noise) — the safety guarantee.
  5. Hybrid wins MORE EV vs. a known-exploitable opponent than pure GTO does.
"""
from __future__ import annotations

from typing import Hashable

from pokerbot.games.base import ExtensiveFormGame
from pokerbot.solvers.exploitability import best_response_value


Strategy = dict[Hashable, dict[Hashable, float]]


def best_response_strategy(
    game: ExtensiveFormGame, opponent_strategy: Strategy, br_player: int
) -> Strategy:
    """Compute the best-response *strategy* (not just its value) against `opponent_strategy`.

    Returns a deterministic strategy {info_set: {action: 1.0}} for `br_player`.

    Implementation: same logic as `best_response_value`, but we record the
    chosen action at each info set instead of (just) returning its value.
    """
    from collections import defaultdict

    # 1. Counterfactual reach to each state (opp + chance only).
    reach: dict = {}

    def walk_reach(state, p: float) -> None:
        reach[state] = p
        if game.is_terminal(state):
            return
        if game.is_chance(state):
            for a, prob in game.chance_outcomes(state):
                walk_reach(game.apply(state, a), p * prob)
            return
        player = game.current_player(state)
        actions = game.legal_actions(state)
        if player == br_player:
            for a in actions:
                walk_reach(game.apply(state, a), p)
        else:
            key = game.info_set_key(state, player)
            opp_strat = opponent_strategy.get(key)
            for a in actions:
                prob = (
                    opp_strat[a] if (opp_strat is not None and a in opp_strat)
                    else 1.0 / len(actions)
                )
                walk_reach(game.apply(state, a), p * prob)

    walk_reach(game.initial_state(), 1.0)

    # 2. Collect states inside each BR info set.
    states_in_info_set: dict = defaultdict(list)

    def walk_collect(state) -> None:
        if game.is_terminal(state):
            return
        if game.is_chance(state):
            for a, _ in game.chance_outcomes(state):
                walk_collect(game.apply(state, a))
            return
        player = game.current_player(state)
        if player == br_player:
            key = game.info_set_key(state, br_player)
            states_in_info_set[key].append(state)
        for a in game.legal_actions(state):
            walk_collect(game.apply(state, a))

    walk_collect(game.initial_state())

    # 3. Resolve BR action at each info set bottom-up.
    br_strategy: dict = {}
    value_cache: dict = {}

    def value_subtree(state):
        if state in value_cache:
            return value_cache[state]

        if game.is_terminal(state):
            v = game.utility(state, br_player)
        elif game.is_chance(state):
            v = 0.0
            for a, prob in game.chance_outcomes(state):
                v += prob * value_subtree(game.apply(state, a))
        else:
            player = game.current_player(state)
            actions = game.legal_actions(state)
            if player != br_player:
                key = game.info_set_key(state, player)
                opp_strat = opponent_strategy.get(key)
                v = 0.0
                for a in actions:
                    prob = (
                        opp_strat[a] if (opp_strat is not None and a in opp_strat)
                        else 1.0 / len(actions)
                    )
                    v += prob * value_subtree(game.apply(state, a))
            else:
                key = game.info_set_key(state, br_player)
                if key not in br_strategy:
                    best_a = None
                    best_total = -float("inf")
                    for a in actions:
                        total = 0.0
                        for s in states_in_info_set[key]:
                            total += reach[s] * value_subtree(game.apply(s, a))
                        if total > best_total:
                            best_total = total
                            best_a = a
                    br_strategy[key] = best_a
                v = value_subtree(game.apply(state, br_strategy[key]))

        value_cache[state] = v
        return v

    value_subtree(game.initial_state())

    # Convert {info_set: chosen_action} to {info_set: {action: prob}}.
    out: Strategy = {}
    for key, chosen in br_strategy.items():
        # Need the action set at this info set.
        actions = states_in_info_set[key]
        legal = game.legal_actions(actions[0])
        out[key] = {a: (1.0 if a == chosen else 0.0) for a in legal}
    return out


def mix_strategies(a: Strategy, b: Strategy, lam: float) -> Strategy:
    """Return (1-lam)*a + lam*b at every shared info set.

    If an info set is in only one of the two, that one is used unmodified.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0,1], got {lam}")
    out: Strategy = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        ad = a.get(k)
        bd = b.get(k)
        if ad is None:
            out[k] = dict(bd) if bd else {}
            continue
        if bd is None:
            out[k] = dict(ad)
            continue
        actions = set(ad.keys()) | set(bd.keys())
        out[k] = {act: (1 - lam) * ad.get(act, 0.0) + lam * bd.get(act, 0.0) for act in actions}
    return out


class HybridPolicy:
    """Wraps GTO blueprint + exploit BR + mixing weight."""

    def __init__(
        self,
        game: ExtensiveFormGame,
        gto_strategy: Strategy,
        player: int,
        base_lambda: float = 0.5,
    ):
        self.game = game
        self.gto = gto_strategy
        self.player = player
        self.base_lambda = base_lambda
        # Computed once we know the opponent's modeled strategy.
        self.exploit_br: Strategy | None = None
        self._lambda = 0.0

    def update_opponent_model(
        self,
        opponent_modeled_strategy: Strategy,
        confidence: float,
        tilt: float = 0.0,
    ) -> None:
        """Set the BR strategy and the mixing weight from the opponent model."""
        self.exploit_br = best_response_strategy(
            self.game, opponent_modeled_strategy, self.player
        )
        # confidence in [0,1], tilt in [0,1]. Tilt reduces our willingness to
        # commit (since the opponent is non-stationary).
        self._lambda = self.base_lambda * confidence * (1.0 - 0.7 * tilt)
        self._lambda = max(0.0, min(1.0, self._lambda))

    @property
    def current_lambda(self) -> float:
        return self._lambda

    def strategy(self) -> Strategy:
        if self.exploit_br is None or self._lambda <= 0.0:
            return self.gto
        return mix_strategies(self.gto, self.exploit_br, self._lambda)

    def action_probs(self, info_set_key: Hashable) -> dict[Hashable, float]:
        return self.strategy().get(info_set_key, {})
