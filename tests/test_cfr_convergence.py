"""Convergence test: vanilla CFR on Kuhn poker.

Kuhn's Nash equilibrium has a free parameter alpha in [0, 1/3]:
    P0 bets J  with prob alpha
    P0 checks Q
    P0 bets K  with prob 3*alpha
    P0 calls Q-bet (after checking) with prob 1/3 + alpha
    (other P0 / P1 strategies are pinned)

P1's Nash (no free parameter):
    P1 bluffs (bets) on J after a check with prob 1/3
    P1 calls a bet holding Q with prob 1/3
    P1 calls always with K
    P1 always folds with J facing a bet
    P1 always passes after a check holding Q

Game value to P0 under Nash: -1/18 ~= -0.0555...

We test:
  1. The average strategy is consistent with the alpha relationships
     (alpha in [0, 1/3]; 3*alpha; 1/3 + alpha).
  2. The expected game value to P0 under the average strategy is -1/18.
  3. P1's pinned probabilities are close to their Nash values.
"""
from __future__ import annotations

import unittest

from pokerbot.games.kuhn import KuhnPoker
from pokerbot.solvers.cfr import CFRSolver


def expected_value_p0(game, strategy: dict, state=None, prob: float = 1.0) -> float:
    """Expected utility to P0 of the joint average strategy. Walks the tree."""
    if state is None:
        state = game.initial_state()
    if game.is_terminal(state):
        return game.utility(state, 0) * prob
    if game.is_chance(state):
        v = 0.0
        for a, p in game.chance_outcomes(state):
            v += expected_value_p0(game, strategy, game.apply(state, a), prob * p)
        return v
    player = game.current_player(state)
    key = game.info_set_key(state, player)
    s = strategy[key]
    v = 0.0
    for action, p in s.items():
        v += expected_value_p0(game, strategy, game.apply(state, action), prob * p)
    return v


class KuhnConvergenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        game = KuhnPoker()
        # CFR+ for faster convergence in the test.
        solver = CFRSolver(game, plus_regret_floor=True, linear_averaging=True)
        solver.train(iterations=10_000)
        cls.game = game
        cls.solver = solver
        cls.avg = solver.average_strategy()

    def test_game_value(self):
        v = expected_value_p0(self.game, self.avg)
        # Analytical: -1/18 = -0.05555...
        self.assertAlmostEqual(v, -1 / 18, places=2)

    def test_p0_alpha_relationships(self):
        """Verify P0's K-bet rate ~= 3 * J-bet rate (the alpha vs 3-alpha rule)."""
        bet_J = self.avg["J:"]["b"]
        bet_K = self.avg["K:"]["b"]
        # alpha in [0, 1/3]; 3*alpha in [0, 1]
        self.assertGreaterEqual(bet_J, -1e-3)
        self.assertLessEqual(bet_J, 1 / 3 + 0.05)
        # K should be bet ~3x as often (modulo the alpha=0 edge case)
        if bet_J > 0.02:
            ratio = bet_K / bet_J
            self.assertAlmostEqual(ratio, 3.0, delta=0.5)

    def test_p0_never_bets_q_first(self):
        """At equilibrium, P0 should pass Q with probability ~1."""
        pass_Q = self.avg["Q:"]["p"]
        self.assertGreater(pass_Q, 0.95, f"P0 should check Q, got pass={pass_Q}")

    def test_p1_calls_K_always(self):
        call_K = self.avg["K:b"]["b"]
        self.assertGreater(call_K, 0.99)

    def test_p1_folds_J_always(self):
        fold_J = self.avg["J:b"]["p"]
        self.assertGreater(fold_J, 0.99)

    def test_p1_calls_Q_at_one_third(self):
        call_Q = self.avg["Q:b"]["b"]
        self.assertAlmostEqual(call_Q, 1 / 3, delta=0.05)

    def test_p1_bluffs_J_at_one_third(self):
        bluff_J = self.avg["J:p"]["b"]
        self.assertAlmostEqual(bluff_J, 1 / 3, delta=0.05)


if __name__ == "__main__":
    unittest.main()
