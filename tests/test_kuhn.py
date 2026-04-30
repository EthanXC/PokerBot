"""Sanity tests for Kuhn poker.

Kuhn has exactly 12 info sets (6 per player: each of {J,Q,K} crossed with
the two histories where that player acts).
"""
from __future__ import annotations

import unittest

from pokerbot.games.base import all_info_sets
from pokerbot.games.kuhn import KuhnPoker


class KuhnStructureTest(unittest.TestCase):
    def setUp(self):
        self.game = KuhnPoker()

    def test_info_set_count(self):
        for player in (0, 1):
            sets = all_info_sets(self.game, player)
            self.assertEqual(
                len(sets), 6,
                f"player {player} should have 6 info sets, got {len(sets)}: {list(sets)}"
            )

    def test_info_set_keys_p0(self):
        sets = all_info_sets(self.game, 0)
        expected = {"J:", "Q:", "K:", "J:pb", "Q:pb", "K:pb"}
        self.assertEqual(set(sets.keys()), expected)

    def test_info_set_keys_p1(self):
        sets = all_info_sets(self.game, 1)
        expected = {"J:p", "Q:p", "K:p", "J:b", "Q:b", "K:b"}
        self.assertEqual(set(sets.keys()), expected)

    def test_zero_sum_at_terminals(self):
        """For every terminal state, u0 + u1 == 0."""
        game = self.game
        seen = 0

        def walk(state):
            nonlocal seen
            if game.is_terminal(state):
                u0 = game.utility(state, 0)
                u1 = game.utility(state, 1)
                self.assertEqual(u0 + u1, 0, f"non-zero-sum at {state}: {u0}+{u1}")
                seen += 1
                return
            if game.is_chance(state):
                for a, _p in game.chance_outcomes(state):
                    walk(game.apply(state, a))
                return
            for a in game.legal_actions(state):
                walk(game.apply(state, a))

        walk(game.initial_state())
        # 6 card deals * 5 terminal histories = 30 leaves
        self.assertEqual(seen, 30)

    def test_chance_probs_sum_to_1(self):
        game = self.game
        s0 = game.initial_state()
        ps = [p for _, p in game.chance_outcomes(s0)]
        self.assertAlmostEqual(sum(ps), 1.0)
        # After dealing one card, the second deal also sums to 1
        s1 = game.apply(s0, (1,))
        ps = [p for _, p in game.chance_outcomes(s1)]
        self.assertAlmostEqual(sum(ps), 1.0)


if __name__ == "__main__":
    unittest.main()
