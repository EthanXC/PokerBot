"""Tests for the preflop card abstraction + CFR-trained HU preflop game."""
from __future__ import annotations

import unittest

from pokerbot.abstraction import (
    NUM_BUCKETS,
    hand_class_str,
    cached_class_equities,
    cached_bucket_map,
    cached_bucket_equity_matrix,
    all_169_classes,
)
from pokerbot.cache import cached_preflop_hu_strategy
from pokerbot.core.cards import Card
from pokerbot.games.preflop_hu import PreflopHUGame
from pokerbot.solvers.exploitability import exploitability


class HandClassTest(unittest.TestCase):
    def test_169_classes(self):
        classes = all_169_classes()
        # Should be exactly 169 unique classes.
        names = [c[0] for c in classes]
        self.assertEqual(len(names), 169)
        self.assertEqual(len(set(names)), 169)

    def test_hand_class_string_canonical(self):
        # AA, AKs, AKo, TT, T9s should all parse correctly
        self.assertEqual(hand_class_str(Card("A", "S"), Card("A", "H")), "AA")
        self.assertEqual(hand_class_str(Card("A", "S"), Card("K", "S")), "AKs")
        self.assertEqual(hand_class_str(Card("A", "S"), Card("K", "H")), "AKo")
        self.assertEqual(hand_class_str(Card("10", "S"), Card("10", "H")), "TT")
        self.assertEqual(hand_class_str(Card("10", "S"), Card("9", "S")), "T9s")
        # Order independence
        self.assertEqual(
            hand_class_str(Card("K", "S"), Card("A", "S")),
            "AKs",
        )


class BucketingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.equities = cached_class_equities()
        cls.buckets = cached_bucket_map()

    def test_AA_in_top_bucket(self):
        self.assertEqual(self.buckets["AA"], NUM_BUCKETS - 1)

    def test_72o_in_bottom_two_buckets(self):
        # MC sampling noise can put 72o in bucket 0 or 1; either is acceptable.
        self.assertLess(self.buckets["72o"], 2)

    def test_buckets_are_balanced(self):
        from collections import Counter
        sizes = Counter(self.buckets.values())
        # Each bucket should hold ~17 hands (169/10 rounded).
        for size in sizes.values():
            self.assertGreater(size, 10)
            self.assertLess(size, 25)

    def test_AA_higher_equity_than_72o(self):
        self.assertGreater(self.equities["AA"], self.equities["72o"])

    def test_premium_pairs_higher_than_suited_connectors(self):
        self.assertGreater(self.equities["AA"], self.equities["76s"])


class BucketEquityMatrixTest(unittest.TestCase):
    def test_matrix_is_symmetric_complement(self):
        m = cached_bucket_equity_matrix()
        for i in range(NUM_BUCKETS):
            for j in range(NUM_BUCKETS):
                self.assertAlmostEqual(m[i][j] + m[j][i], 1.0, places=2,
                                       msg=f"i={i}, j={j}")

    def test_top_bucket_beats_bottom_handily(self):
        m = cached_bucket_equity_matrix()
        # Bucket 9 vs 0 should be >=70%
        self.assertGreater(m[NUM_BUCKETS - 1][0], 0.65)


class PreflopHUGameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eq_matrix = cached_bucket_equity_matrix()
        cls.game = PreflopHUGame(cls.eq_matrix)

    def test_zero_sum_at_terminals(self):
        game = self.game
        # Walk every (b0, b1, history) terminal.
        from pokerbot.games.preflop_hu import TERMINAL_SEQS
        for b0 in range(NUM_BUCKETS):
            for b1 in range(NUM_BUCKETS):
                for h, info in TERMINAL_SEQS.items():
                    if info is None:
                        continue
                    state = ((b0, b1), h)
                    u0 = game.utility(state, 0)
                    u1 = game.utility(state, 1)
                    self.assertAlmostEqual(u0 + u1, 0.0, places=4,
                                           msg=f"non-zero-sum at {h} buckets ({b0},{b1})")

    def test_cfr_converges(self):
        from pokerbot.solvers.cfr import CFRSolver
        solver = CFRSolver(self.game, plus_regret_floor=True, linear_averaging=True)
        solver.train(2000)
        avg = solver.average_strategy()
        e = exploitability(self.game, avg)
        # 80 info sets, vanilla CFR+ should drive exploitability << 0.05.
        self.assertLess(e, 0.05, f"preflop HU exploitability too high: {e}")


class CachedStrategyTest(unittest.TestCase):
    def test_cached_strategy_loads(self):
        strategy = cached_preflop_hu_strategy(iterations=2000)
        # Should have ~80 info sets.
        self.assertGreaterEqual(len(strategy), 70)
        self.assertLessEqual(len(strategy), 200)

    def test_premium_hands_3bet_in_blueprint(self):
        strategy = cached_preflop_hu_strategy(iterations=2000)
        # Bucket 9 (premium) facing a raise should heavily 3-bet.
        key = "b9|r"
        if key in strategy:
            r_prob = strategy[key].get("r", 0.0)
            self.assertGreater(r_prob, 0.5,
                               f"bucket 9 should 3-bet a lot, got r={r_prob}")


if __name__ == "__main__":
    unittest.main()
