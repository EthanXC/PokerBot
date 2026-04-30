"""Tests for postflop hand bucketing + heads-up postflop CFR."""
from __future__ import annotations

import unittest

from pokerbot.abstraction import (
    NUM_POSTFLOP_BUCKETS,
    bucket_for_strength,
    bucket_for_hole_and_board,
    bucket_name,
    cached_postflop_bucket_equities,
)
from pokerbot.cache import cached_postflop_hu_strategy
from pokerbot.core.cards import Card
from pokerbot.games.postflop_hu import PostflopHUGame, TERMINAL_SEQS
from pokerbot.solvers.exploitability import exploitability


class PostflopBucketTest(unittest.TestCase):
    def test_5_bucket_count(self):
        self.assertEqual(NUM_POSTFLOP_BUCKETS, 5)

    def test_bucket_for_strength_boundaries(self):
        self.assertEqual(bucket_for_strength(0.10), 0)   # air
        self.assertEqual(bucket_for_strength(0.25), 1)   # weak
        self.assertEqual(bucket_for_strength(0.50), 2)   # medium
        self.assertEqual(bucket_for_strength(0.70), 3)   # strong
        self.assertEqual(bucket_for_strength(0.90), 4)   # nuts

    def test_bucket_names(self):
        self.assertEqual(bucket_name(0), "air")
        self.assertEqual(bucket_name(4), "nuts")

    def test_top_pair_is_at_least_medium(self):
        # K-high board, holding KQ → top pair → at least medium (bucket 2).
        hole = (Card("K", "S"), Card("Q", "D"))
        board = (Card("K", "H"), Card("7", "C"), Card("3", "D"))
        b = bucket_for_hole_and_board(hole, board)
        self.assertGreaterEqual(b, 2, f"top pair should be >=medium (>=2), got {b}")

    def test_set_is_strong_or_nuts(self):
        # KK on K-7-3 board = set of kings → strong/nuts
        hole = (Card("K", "S"), Card("K", "D"))
        board = (Card("K", "H"), Card("7", "C"), Card("3", "D"))
        b = bucket_for_hole_and_board(hole, board)
        self.assertGreaterEqual(b, 3, f"set should be strong+ (>=3), got {b}")

    def test_air_on_dry_board(self):
        # 23 holding on a K-7-3 board: nothing
        hole = (Card("2", "C"), Card("3", "S"))  # would pair the 3
        # Use 24 instead
        hole = (Card("2", "C"), Card("4", "S"))
        board = (Card("K", "H"), Card("9", "D"), Card("7", "C"))
        b = bucket_for_hole_and_board(hole, board)
        self.assertLess(b, 2, f"air should be 0-1, got {b}")


class PostflopEquityMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = cached_postflop_bucket_equities()

    def test_diagonal_is_half(self):
        for i in range(NUM_POSTFLOP_BUCKETS):
            self.assertAlmostEqual(self.matrix[i][i], 0.5, places=4)

    def test_nuts_beats_air(self):
        self.assertGreater(self.matrix[4][0], 0.85)

    def test_strong_beats_weak(self):
        self.assertGreater(self.matrix[3][1], 0.75)


class PostflopHUGameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eq_matrix = cached_postflop_bucket_equities()
        cls.game = PostflopHUGame(cls.eq_matrix)

    def test_zero_sum_terminals(self):
        for b0 in range(NUM_POSTFLOP_BUCKETS):
            for b1 in range(NUM_POSTFLOP_BUCKETS):
                for h, info in TERMINAL_SEQS.items():
                    if info is None:
                        continue
                    state = ((b0, b1), h)
                    u0 = self.game.utility(state, 0)
                    u1 = self.game.utility(state, 1)
                    self.assertAlmostEqual(
                        u0 + u1, 0.0, places=4,
                        msg=f"non-zero-sum at history={h} buckets=({b0},{b1})"
                    )

    def test_cfr_converges(self):
        from pokerbot.solvers.cfr import CFRSolver
        solver = CFRSolver(self.game, plus_regret_floor=True, linear_averaging=True)
        solver.train(2000)
        avg = solver.average_strategy()
        e = exploitability(self.game, avg)
        self.assertLess(e, 0.05, f"postflop HU exploitability too high: {e}")


class CachedStrategyTest(unittest.TestCase):
    def test_cached_strategy_loads(self):
        strategy = cached_postflop_hu_strategy(iterations=2000)
        # ~40 info sets expected
        self.assertGreaterEqual(len(strategy), 25)
        self.assertLessEqual(len(strategy), 100)

    def test_nuts_facing_bet_continues(self):
        strategy = cached_postflop_hu_strategy(iterations=2000)
        # Bucket 4 facing a bet must call or raise (never fold 100%).
        key = "pf4|b"
        if key in strategy:
            d = strategy[key]
            fold_p = d.get("f", 0.0)
            self.assertLess(fold_p, 0.5,
                            f"nuts shouldn't fold to a bet; got fold={fold_p}")


if __name__ == "__main__":
    unittest.main()
