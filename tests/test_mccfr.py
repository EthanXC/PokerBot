"""MCCFR tests.

Stochastic algorithm — we use seeded RNG and looser tolerances than vanilla CFR.
"""
from __future__ import annotations

import random
import unittest

from pokerbot.games.kuhn import KuhnPoker
from pokerbot.games.leduc import LeducPoker
from pokerbot.solvers.mccfr import MCCFRSolver
from pokerbot.solvers.exploitability import exploitability


class MCCFRKuhnTest(unittest.TestCase):
    def test_kuhn_low_exploitability(self):
        game = KuhnPoker()
        rng = random.Random(42)
        solver = MCCFRSolver(game, plus_regret_floor=True, linear_averaging=True, rng=rng)
        solver.train(50_000)
        e = exploitability(game, solver.average_strategy())
        # Looser bound than vanilla CFR because of sampling variance.
        self.assertLess(e, 0.05, f"MCCFR Kuhn exploitability too high: {e}")


class MCCFRLeducTest(unittest.TestCase):
    def test_leduc_exploitability_decreases(self):
        game = LeducPoker()
        rng = random.Random(0)
        solver = MCCFRSolver(game, plus_regret_floor=True, linear_averaging=True, rng=rng)

        chk = []
        for _ in range(3):
            solver.train(2_000)
            e = exploitability(game, solver.average_strategy())
            chk.append(e)

        # Allow some noise; require global trend down across the 3 checkpoints.
        self.assertLess(chk[-1], chk[0], f"MCCFR Leduc didn't improve: {chk}")
        self.assertLess(chk[-1], 0.5, f"final expl too high: {chk}")


if __name__ == "__main__":
    unittest.main()
