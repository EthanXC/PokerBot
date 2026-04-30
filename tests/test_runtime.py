"""Tests for the runtime subpackage: simulator, session runner, feature extraction."""
from __future__ import annotations

import random
import unittest

import numpy as np

from pokerbot.games.leduc import card_rank
from pokerbot.runtime import (
    LeducHeuristicPlayer,
    PROFILES,
    make_player,
    run_session,
    extract_bluff_examples,
    extract_tilt_examples,
    build_bluff_dataset,
    build_tilt_dataset,
)
from pokerbot.runtime.heuristic_player import hand_strength
from pokerbot.runtime.features import _is_value_or_bluff_for_bettor


class HandStrengthTest(unittest.TestCase):
    def test_round1_strength(self):
        # No board: K=strong, Q=marginal, J=weak.
        # Card encoding: 0,1=J; 2,3=Q; 4,5=K
        self.assertEqual(hand_strength(0, None), "weak")     # J
        self.assertEqual(hand_strength(2, None), "marginal") # Q
        self.assertEqual(hand_strength(4, None), "strong")   # K

    def test_round2_pair_is_strong(self):
        # K with K-board => pair, strong
        self.assertEqual(hand_strength(4, 5), "strong")  # K0 with K1
        # Q with Q-board
        self.assertEqual(hand_strength(2, 3), "strong")
        # J with J-board
        self.assertEqual(hand_strength(0, 1), "strong")

    def test_round2_overcard_is_marginal(self):
        # K with Q-board (no pair, but overcard)
        self.assertEqual(hand_strength(4, 2), "marginal")

    def test_round2_undercard_is_weak(self):
        # J with Q-board
        self.assertEqual(hand_strength(0, 2), "weak")


class HeuristicPlayerTest(unittest.TestCase):
    def test_tight_passive_rarely_bluffs(self):
        rng = random.Random(0)
        p = make_player("tight_passive", rng=rng)
        # The profile has bluff_freq=0.05, so over 200 trials with weak hand,
        # bet rate should be around 5%.
        from pokerbot.games.leduc import LeducPoker
        game = LeducPoker()
        # Make a state where the player is acting first round 1 with J0:
        state = ((0, 4), "")  # P0=J0, P1=K0; P0 to act first
        bets = 0
        for _ in range(500):
            a = p.decide(game, state, ["k", "b"], my_player=0)
            if a == "b":
                bets += 1
        rate = bets / 500
        self.assertLess(rate, 0.15, f"tight_passive bluff rate too high: {rate}")

    def test_maniac_bets_a_lot(self):
        rng = random.Random(0)
        p = make_player("maniac", rng=rng)
        from pokerbot.games.leduc import LeducPoker
        game = LeducPoker()
        state = ((0, 4), "")  # weak hand
        bets = 0
        for _ in range(500):
            a = p.decide(game, state, ["k", "b"], my_player=0)
            if a == "b":
                bets += 1
        rate = bets / 500
        self.assertGreater(rate, 0.45, f"maniac bluff rate too low: {rate}")

    def test_tilt_kicks_in_after_losses(self):
        p = make_player("tilt_prone", rng=random.Random(0))
        self.assertFalse(p.is_tilted())
        for _ in range(3):
            p.observe_result(-3.0)
        self.assertTrue(p.is_tilted())


class SessionRunnerTest(unittest.TestCase):
    def test_session_produces_valid_traces(self):
        p0 = make_player("loose_aggressive", rng=random.Random(1))
        p1 = make_player("tight_passive", rng=random.Random(2))
        traces = run_session(p0, p1, 50, rng=random.Random(3))
        self.assertEqual(len(traces), 50)
        for t in traces:
            # Zero-sum
            self.assertAlmostEqual(t.p0_net + t.p1_net, 0.0, places=9)
            # Holes are integers in [0,5]
            self.assertEqual(len(t.hole_cards), 2)
            self.assertTrue(all(0 <= c <= 5 for c in t.hole_cards))
            # If went to showdown, board is dealt
            if t.went_to_showdown:
                self.assertIsNotNone(t.board_card)
            # Actions are recorded
            self.assertGreater(len(t.actions), 0)

    def test_winner_matches_payoff_sign(self):
        p0 = make_player("calling_station", rng=random.Random(11))
        p1 = make_player("calling_station", rng=random.Random(12))
        traces = run_session(p0, p1, 100, rng=random.Random(13))
        for t in traces:
            if t.winner == 0:
                self.assertGreater(t.p0_net, 0)
            elif t.winner == 1:
                self.assertGreater(t.p1_net, 0)
            else:
                self.assertEqual(t.p0_net, 0)


class BluffLabelTest(unittest.TestCase):
    def test_round1_K_is_value(self):
        # K bettor preflop = strongest hand category, label = 0 (value)
        self.assertEqual(_is_value_or_bluff_for_bettor(4, None), 0)
        self.assertEqual(_is_value_or_bluff_for_bettor(5, None), 0)

    def test_round1_J_is_bluff(self):
        self.assertEqual(_is_value_or_bluff_for_bettor(0, None), 1)
        self.assertEqual(_is_value_or_bluff_for_bettor(1, None), 1)

    def test_round2_pair_is_value(self):
        # K0 with K1 board = pair = value
        self.assertEqual(_is_value_or_bluff_for_bettor(4, 5), 0)

    def test_round2_no_pair_is_bluff(self):
        # K with Q board: no pair, even though overcard, we call it a bluff
        self.assertEqual(_is_value_or_bluff_for_bettor(4, 2), 1)


class FeatureExtractionTest(unittest.TestCase):
    def test_bluff_examples_have_correct_shape(self):
        p0 = make_player("loose_aggressive", rng=random.Random(11))
        p1 = make_player("tight_passive", rng=random.Random(12))
        traces = run_session(p0, p1, 200, rng=random.Random(13))
        X, y = extract_bluff_examples(traces)
        self.assertGreater(len(X), 50)
        self.assertEqual(X.shape[1], 12)
        # Labels are binary
        self.assertTrue(set(np.unique(y)).issubset({0, 1}))
        # Both classes present
        self.assertGreater((y == 1).sum(), 0)
        self.assertGreater((y == 0).sum(), 0)

    def test_tilt_examples_have_correct_shape(self):
        p0 = make_player("tilt_prone", rng=random.Random(1))
        p1 = make_player("calling_station", rng=random.Random(2))
        traces = run_session(p0, p1, 300, rng=random.Random(3))
        X, y = extract_tilt_examples(traces)
        self.assertGreater(len(X), 100)
        self.assertEqual(X.shape[1], 7)
        self.assertTrue(set(np.unique(y)).issubset({0, 1}))


class DataGenerationTest(unittest.TestCase):
    def test_build_bluff_dataset(self):
        X, y = build_bluff_dataset(n_hands_per_pairing=50, seed=0)
        self.assertGreater(len(X), 100)
        self.assertEqual(X.shape[1], 12)
        # Reasonable class balance
        bluff_rate = (y == 1).mean()
        self.assertGreater(bluff_rate, 0.1)
        self.assertLess(bluff_rate, 0.9)

    def test_build_tilt_dataset(self):
        X, y = build_tilt_dataset(n_hands_per_pairing=50, seed=0)
        self.assertGreater(len(X), 100)
        self.assertEqual(X.shape[1], 7)
        # Tilt is rarer; just confirm both classes appear
        self.assertGreater((y == 1).sum(), 0)
        self.assertGreater((y == 0).sum(), 0)


if __name__ == "__main__":
    unittest.main()
