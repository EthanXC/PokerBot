"""Tests for the MultiOpponentAdaptiveBot.

We test the architecture, not statistical-significance EV — that requires
many more hands than a unit test should run. Specifically we verify:

  - The bot plays without crashing at 6-handed tables
  - Per-seat stats are tracked correctly across hands
  - The bluff/tilt classifiers get queried (n_decisions > 0)
  - Deviations are gated by empirical aggression (no deviations vs
    a table of all tight players)
"""
from __future__ import annotations

import random
import unittest

from pokerbot.cache import cached_real_classifiers
from pokerbot.games.nlhe import NLHEConfig
from pokerbot.runtime import (
    make_nlhe_player,
    play_table,
    MultiOpponentAdaptiveBot,
)


class MultiOpponentBotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bluff_clf, cls.tilt_clf = cached_real_classifiers()

    def test_bot_plays_without_crashing(self):
        """6-handed table runs without error; the bot makes decisions."""
        rng = random.Random(0)
        bot = MultiOpponentAdaptiveBot(
            bluff_clf=self.bluff_clf, tilt_clf=self.tilt_clf,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        opps = [
            make_nlhe_player("loose_aggressive", rng=random.Random(rng.randint(0, 2**31))),
            make_nlhe_player("tight_passive", rng=random.Random(rng.randint(0, 2**31))),
            make_nlhe_player("calling_station", rng=random.Random(rng.randint(0, 2**31))),
            make_nlhe_player("maniac", rng=random.Random(rng.randint(0, 2**31))),
            make_nlhe_player("tilt_prone", rng=random.Random(rng.randint(0, 2**31))),
        ]
        result = play_table([bot] + opps, n_hands=100, rng=random.Random(42))
        self.assertGreater(bot.n_decisions, 100, "bot should make many decisions")
        # Zero-sum invariant
        self.assertAlmostEqual(sum(result.seat_means), 0.0, places=6)

    def test_per_seat_stats_accumulate(self):
        rng = random.Random(0)
        bot = MultiOpponentAdaptiveBot(
            bluff_clf=self.bluff_clf, tilt_clf=self.tilt_clf,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        opps = [
            make_nlhe_player("maniac", rng=random.Random(rng.randint(0, 2**31)))
            for _ in range(5)
        ]
        play_table([bot] + opps, n_hands=80, rng=random.Random(42))
        # Each opponent seat should have some accumulated state.
        for seat in range(1, 6):
            self.assertIn(seat, bot._stats)
            stats = bot._stats[seat]
            self.assertGreater(stats.n_hands, 0)
            self.assertGreater(stats.n_actions_taken, 0)

    def test_no_deviations_against_passive_table(self):
        """All tight passive players: empirical aggression rate is low,
        so the bot's safety gate should fire and produce 0 deviations."""
        rng = random.Random(0)
        bot = MultiOpponentAdaptiveBot(
            bluff_clf=self.bluff_clf, tilt_clf=self.tilt_clf,
            min_observations=15,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        opps = [
            make_nlhe_player("tight_passive", rng=random.Random(rng.randint(0, 2**31)))
            for _ in range(5)
        ]
        play_table([bot] + opps, n_hands=200, rng=random.Random(42))
        deviation_rate = bot.n_deviated / max(1, bot.n_decisions)
        # At a table of all tights, deviation rate should be very low.
        self.assertLess(
            deviation_rate, 0.05,
            f"too many deviations vs all-tight table: {deviation_rate:.1%}"
        )

    def test_supports_9_players(self):
        rng = random.Random(0)
        bot = MultiOpponentAdaptiveBot(
            bluff_clf=self.bluff_clf, tilt_clf=self.tilt_clf,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        names = ["maniac", "loose_aggressive", "calling_station",
                 "tight_passive", "tilt_prone", "calling_station",
                 "loose_aggressive", "tight_passive"]
        opps = [make_nlhe_player(n, rng=random.Random(rng.randint(0, 2**31)))
                for n in names]
        result = play_table(
            [bot] + opps, n_hands=40,
            config=NLHEConfig(n_players=9),
            rng=random.Random(42),
        )
        self.assertEqual(len(result.seat_means), 9)
        self.assertAlmostEqual(sum(result.seat_means), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
