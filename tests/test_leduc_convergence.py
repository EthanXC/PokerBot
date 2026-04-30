"""Convergence test on Leduc poker.

Unlike Kuhn, Leduc doesn't have a tidy closed-form Nash. The standard test
is just "exploitability shrinks toward zero monotonically".

Slow test (~30s on a laptop) — we run it as a single test method.
"""
from __future__ import annotations

import unittest

from pokerbot.games.leduc import LeducPoker
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.solvers.exploitability import exploitability


class LeducConvergenceTest(unittest.TestCase):
    def test_exploitability_decreases(self):
        game = LeducPoker()
        solver = CFRSolver(game, plus_regret_floor=True, linear_averaging=True)

        # Sample exploitability at 3 checkpoints. Each must be lower than the prior.
        checkpoints = []
        for chunk in range(3):
            solver.train(iterations=50)
            avg = solver.average_strategy()
            e = exploitability(game, avg)
            checkpoints.append(e)

        # Strict monotonicity (with tolerance for noise).
        self.assertGreater(
            checkpoints[0], checkpoints[1],
            f"expl should decrease from iter 50 to 100: {checkpoints}"
        )
        self.assertGreater(
            checkpoints[1], checkpoints[2],
            f"expl should decrease from iter 100 to 150: {checkpoints}"
        )
        # Final exploitability should be well below 0.5 chips/game (~125 mbb/g).
        self.assertLess(
            checkpoints[-1], 0.2,
            f"after 150 iterations expl should be <0.2 chips, got {checkpoints[-1]}"
        )


if __name__ == "__main__":
    unittest.main()
