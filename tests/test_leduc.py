"""Sanity tests for Leduc poker.

Validates:
  - Game tree is zero-sum at every terminal.
  - Info-set count is in the expected ballpark (~hundreds per player).
  - Specific hand-coded scenarios produce correct payoffs.
"""
from __future__ import annotations

import unittest

from pokerbot.games.base import all_info_sets
from pokerbot.games.leduc import LeducPoker, BET_SIZE, ANTE


class LeducStructureTest(unittest.TestCase):
    def setUp(self):
        self.game = LeducPoker()

    def test_zero_sum_at_terminals(self):
        game = self.game
        terminals = 0
        non_zero = 0

        def walk(state):
            nonlocal terminals, non_zero
            if game.is_terminal(state):
                u0 = game.utility(state, 0)
                u1 = game.utility(state, 1)
                if abs(u0 + u1) > 1e-9:
                    non_zero += 1
                terminals += 1
                return
            if game.is_chance(state):
                for a, _ in game.chance_outcomes(state):
                    walk(game.apply(state, a))
                return
            for a in game.legal_actions(state):
                walk(game.apply(state, a))

        walk(game.initial_state())
        self.assertEqual(non_zero, 0, f"{non_zero} terminals were not zero-sum")
        self.assertGreater(terminals, 100, "expected hundreds of terminal states")

    def test_info_set_counts_in_ballpark(self):
        for player in (0, 1):
            sets = all_info_sets(self.game, player)
            # Leduc has ~144 info sets per player at the standard rule set used here.
            # Bound generously: between 50 and 1500 per player.
            self.assertGreater(
                len(sets), 50,
                f"player {player} info set count too low: {len(sets)}"
            )
            self.assertLess(
                len(sets), 1500,
                f"player {player} info set count too high: {len(sets)}"
            )

    def test_chance_outcomes_complete(self):
        """At every chance node, probabilities sum to 1."""
        game = self.game

        def walk(state):
            if game.is_terminal(state):
                return
            if game.is_chance(state):
                outcomes = game.chance_outcomes(state)
                total = sum(p for _, p in outcomes)
                self.assertAlmostEqual(
                    total, 1.0, places=9,
                    msg=f"chance outcomes don't sum to 1 at {state}: {total}"
                )
                for a, _ in outcomes:
                    walk(game.apply(state, a))
                return
            for a in game.legal_actions(state):
                walk(game.apply(state, a))

        walk(game.initial_state())

    def test_check_check_check_check_pot(self):
        """Both players check both rounds. Showdown for the antes."""
        game = self.game
        # Manually construct: P0=J0, P1=Q0, board=K0
        s = ((0, 2, 4), "kk/kk")
        # Apply trailing '/' since round 2 just closed
        # Actually we need to step through apply to get the right history.
        s0 = game.initial_state()
        s = game.apply(s0, (0,))      # P0 dealt J0
        s = game.apply(s, (2,))       # P1 dealt Q0
        s = game.apply(s, "k")        # P0 check
        s = game.apply(s, "k")        # P1 check, round closes; '/' appended
        s = game.apply(s, (4,))       # board K0
        s = game.apply(s, "k")        # P0 check
        s = game.apply(s, "k")        # P1 check, terminal
        self.assertTrue(game.is_terminal(s), f"Expected terminal, got {s}")
        # P1 has Q (rank 1), board is K (rank 2), P0 has J (rank 0). No pairs.
        # P1 (Q) > P0 (J) at high card. P1 wins.
        u0 = game.utility(s, 0)
        u1 = game.utility(s, 1)
        # Both contributed only the ante. Pot = 2. Loser nets -1, winner +1.
        self.assertEqual(u0, -1)
        self.assertEqual(u1, +1)

    def test_round_one_fold_payoffs(self):
        """P0 bets, P1 folds: P0 wins ante from P1."""
        game = self.game
        s = game.initial_state()
        s = game.apply(s, (0,))   # P0=J0
        s = game.apply(s, (2,))   # P1=Q0
        s = game.apply(s, "b")    # P0 bets 2
        s = game.apply(s, "f")    # P1 folds
        self.assertTrue(game.is_terminal(s))
        # P0 contrib = ante + bet = 1 + 2 = 3. P1 contrib = ante = 1. Pot = 4.
        # Winner (P0) net = pot - contrib_p0 = 4 - 3 = 1.
        # Loser (P1) net = -contrib_p1 = -1.
        self.assertEqual(game.utility(s, 0), 1)
        self.assertEqual(game.utility(s, 1), -1)

    def test_pair_beats_high_card_at_showdown(self):
        """P0 pairs the board, P1 doesn't. Both check down. P0 wins."""
        game = self.game
        s = game.initial_state()
        s = game.apply(s, (4,))   # P0=K0
        s = game.apply(s, (2,))   # P1=Q0
        s = game.apply(s, "k")
        s = game.apply(s, "k")
        s = game.apply(s, (5,))   # board=K1 -> P0 pairs Ks
        s = game.apply(s, "k")
        s = game.apply(s, "k")
        self.assertTrue(game.is_terminal(s))
        self.assertEqual(game.utility(s, 0), +1)
        self.assertEqual(game.utility(s, 1), -1)

    def test_raise_cap(self):
        """At most 2 raises per round (a 'bet' counts as the first raise)."""
        game = self.game
        s = game.initial_state()
        s = game.apply(s, (0,))
        s = game.apply(s, (2,))
        s = game.apply(s, "b")    # P0 bets (1st 'raise')
        s = game.apply(s, "r")    # P1 raises (2nd 'raise')
        actions = game.legal_actions(s)
        # Cap reached -> only fold/call allowed.
        self.assertIn("f", actions)
        self.assertIn("c", actions)
        self.assertNotIn("r", actions)


if __name__ == "__main__":
    unittest.main()
