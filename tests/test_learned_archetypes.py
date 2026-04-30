"""Tests for the learned-archetype model and its bot integration."""
from __future__ import annotations

import random
import unittest

import numpy as np

from pokerbot.cache import (
    cached_real_classifiers,
    cached_postflop_hu_strategy,
    cached_learned_archetypes,
)
from pokerbot.games.nlhe import NLHEConfig
from pokerbot.opponent import LearnedArchetypes, STAT_NAMES
from pokerbot.runtime import (
    play_table,
    MultiOpponentAdaptiveBot,
    make_nlhe_player,
)


class LearnedArchetypesShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = cached_learned_archetypes(n_sessions=80, hands_per_session=60)

    def test_correct_number_of_components(self):
        self.assertEqual(self.arch.n_components, 5)

    def test_cluster_means_have_correct_dim(self):
        self.assertEqual(self.arch.cluster_means.shape, (5, len(STAT_NAMES)))

    def test_cluster_weights_sum_to_one(self):
        self.assertAlmostEqual(float(self.arch.cluster_weights.sum()), 1.0, places=4)

    def test_responsibilities_sum_to_one(self):
        x = np.array([0.3, 0.2, 0.55, 0.4])
        r = self.arch.responsibility(x)
        self.assertAlmostEqual(float(r.sum()), 1.0, places=4)
        self.assertEqual(r.shape, (5,))

    def test_exploitability_per_cluster_is_in_unit_interval(self):
        scores = self.arch.exploitability_score_per_cluster()
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


class ExploitabilitySemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = cached_learned_archetypes(n_sessions=80, hands_per_session=60)

    def test_aggressive_player_more_exploitable_than_nit(self):
        """A high-AGG player should score higher than a low-AGG one."""
        # Synthesize stat vectors (out-of-distribution, but the GMM will still
        # produce posterior responsibilities, and we expect ordering to hold).
        nit = np.array([0.18, 0.12, 0.05, 0.20])      # tight passive
        maniac = np.array([0.55, 0.45, 0.45, 0.40])   # loose aggressive
        score_nit = self.arch.opponent_exploitability(nit)
        score_maniac = self.arch.opponent_exploitability(maniac)
        self.assertGreater(
            score_maniac, score_nit,
            f"maniac ({score_maniac:.3f}) should outscore nit ({score_nit:.3f})"
        )


class BotIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bluff_clf, cls.tilt_clf = cached_real_classifiers()
        cls.postflop_strategy = cached_postflop_hu_strategy()
        cls.archetypes = cached_learned_archetypes(n_sessions=80, hands_per_session=60)

    def test_bot_with_archetypes_runs(self):
        rng = random.Random(0)
        bot = MultiOpponentAdaptiveBot(
            bluff_clf=self.bluff_clf, tilt_clf=self.tilt_clf,
            use_preflop_chart=True,
            postflop_hu_strategy=self.postflop_strategy,
            learned_archetypes=self.archetypes,
            rng=random.Random(rng.randint(0, 2 ** 31)),
        )
        opp = make_nlhe_player("loose_aggressive", rng=random.Random(7))
        result = play_table(
            [bot, opp], n_hands=200,
            config=NLHEConfig(n_players=2),
            rng=random.Random(42),
        )
        # Sanity: zero-sum, bot makes decisions.
        self.assertAlmostEqual(sum(result.seat_means), 0.0, places=6)
        self.assertGreater(bot.n_decisions, 100)

    def test_bot_no_regression_with_archetypes(self):
        """Bot WITH learned archetypes should perform within noise of bot WITHOUT."""
        rng = random.Random(0)
        # Without archetypes
        bot_a = MultiOpponentAdaptiveBot(
            bluff_clf=self.bluff_clf, tilt_clf=self.tilt_clf,
            use_preflop_chart=True,
            postflop_hu_strategy=self.postflop_strategy,
            learned_archetypes=None,
            rng=random.Random(rng.randint(0, 2 ** 31)),
        )
        opp_a = make_nlhe_player("calling_station", rng=random.Random(7))
        result_a = play_table(
            [bot_a, opp_a], n_hands=400,
            config=NLHEConfig(n_players=2),
            rng=random.Random(42),
        )

        # With archetypes
        rng = random.Random(0)
        bot_b = MultiOpponentAdaptiveBot(
            bluff_clf=self.bluff_clf, tilt_clf=self.tilt_clf,
            use_preflop_chart=True,
            postflop_hu_strategy=self.postflop_strategy,
            learned_archetypes=self.archetypes,
            rng=random.Random(rng.randint(0, 2 ** 31)),
        )
        opp_b = make_nlhe_player("calling_station", rng=random.Random(7))
        result_b = play_table(
            [bot_b, opp_b], n_hands=400,
            config=NLHEConfig(n_players=2),
            rng=random.Random(42),
        )

        # 95% CI on a 400-hand match is roughly ±2 chips/hand, so the gap
        # between the two should be much smaller than that.
        gap = abs(result_a.seat_means[0] - result_b.seat_means[0])
        self.assertLess(
            gap, 3.0,
            f"adding archetypes shouldn't move EV by more than noise; "
            f"got A={result_a.seat_means[0]:+.3f}, B={result_b.seat_means[0]:+.3f}"
        )


class StatVectorTest(unittest.TestCase):
    """The bot's _opponent_stat_vector should produce sensible numbers."""

    @classmethod
    def setUpClass(cls):
        cls.bluff_clf, cls.tilt_clf = cached_real_classifiers()

    def test_stat_vector_against_calling_station(self):
        rng = random.Random(0)
        bot = MultiOpponentAdaptiveBot(
            bluff_clf=self.bluff_clf, tilt_clf=self.tilt_clf,
            rng=random.Random(rng.randint(0, 2 ** 31)),
        )
        opp = make_nlhe_player("calling_station", rng=random.Random(7))
        play_table(
            [bot, opp], n_hands=200,
            config=NLHEConfig(n_players=2),
            rng=random.Random(42),
        )
        sv = bot._opponent_stat_vector(1)
        self.assertIsNotNone(sv)
        # Calling station has high WTSD (they call down to showdown often)
        # AND low aggression.
        vpip, pfr, agg, wtsd = sv
        self.assertGreater(wtsd, 0.3,
                           f"calling station should reach showdown often, got WTSD={wtsd}")


if __name__ == "__main__":
    unittest.main()
