"""Opponent modeling tests.

We simulate hands matching each archetype's profile and verify the model's
posterior assigns the highest probability to that archetype.
"""
from __future__ import annotations

import random
import unittest

from pokerbot.opponent.stats import BetaStat, OpponentStats
from pokerbot.opponent.archetypes import ArchetypeModel, ARCHETYPES


def fill_stats(rng: random.Random, profile: dict, hands: int = 500) -> OpponentStats:
    """Generate observed stats by sampling from a profile of true frequencies."""
    s = OpponentStats(name="syn")
    for _ in range(hands):
        # Each "hand" we observe one opportunity for each stat.
        s.vpip.observe(rng.random() < profile["VPIP"])
        s.pfr.observe(rng.random() < profile["PFR"])
        s.three_bet.observe(rng.random() < profile.get("3BET", 0.05))
        s.fold_to_3bet.observe(rng.random() < profile["F3B"])
        s.cbet.observe(rng.random() < profile.get("CBET", 0.55))
        s.fold_to_cbet.observe(rng.random() < profile.get("F2C", 0.45))
        s.wtsd.observe(rng.random() < profile["WTSD"])
        s.aggression.observe(rng.random() < profile["AGG"])
        s.hands_played += 1
        s.observe_voluntary_action(rng.random() < profile["VPIP"])
    return s


class ArchetypeRecognitionTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(2026)
        self.model = ArchetypeModel()
        # Profiles to simulate, taken from the archetype canonical means.
        self.profiles = {a.name: a.means for a in ARCHETYPES}

    def test_recognizes_fish(self):
        s = fill_stats(self.rng, self.profiles["fish"], hands=400)
        name, p = self.model.most_likely(s)
        self.assertEqual(name, "fish", f"posterior: {self.model.posterior(s)}")
        self.assertGreater(p, 0.5)

    def test_recognizes_nit(self):
        s = fill_stats(self.rng, self.profiles["nit"], hands=400)
        name, _ = self.model.most_likely(s)
        self.assertEqual(name, "nit")

    def test_recognizes_maniac(self):
        s = fill_stats(self.rng, self.profiles["maniac"], hands=400)
        name, _ = self.model.most_likely(s)
        self.assertEqual(name, "maniac")

    def test_low_data_falls_back_to_prior(self):
        s = OpponentStats()
        post = self.model.posterior(s)
        # With no data, posterior should equal the prior (within tolerance).
        prior_sum = sum(a.prior for a in ARCHETYPES)
        for arch in ARCHETYPES:
            expected = arch.prior / prior_sum
            self.assertAlmostEqual(post[arch.name], expected, places=4)


class TiltDetectionTest(unittest.TestCase):
    def test_steady_player_no_tilt(self):
        s = OpponentStats()
        for _ in range(50):
            s.vpip.observe(True)
            s.observe_voluntary_action(True)
            s.observe_hand_result(0.5)  # small wins
            s.hands_played += 1
        for _ in range(50):
            s.vpip.observe(False)
            s.observe_voluntary_action(False)
            s.observe_hand_result(-0.5)  # small losses
            s.hands_played += 1
        self.assertLess(s.tilt_score(), 0.3)

    def test_recent_big_loss_raises_tilt_score(self):
        s = OpponentStats()
        # Build long-term VPIP at 25%
        for i in range(200):
            voluntary = i % 4 == 0  # 25%
            s.vpip.observe(voluntary)
        # Recent 20 hands: big losses + spike VPIP to 80%
        for i in range(20):
            s.observe_hand_result(-3.0)
            s.observe_voluntary_action(i % 5 != 0)  # 80%
        self.assertGreater(s.tilt_score(), 0.4,
                           f"expected tilt detection, got {s.tilt_score()}")

    def test_tilt_score_zero_with_no_data(self):
        s = OpponentStats()
        self.assertEqual(s.tilt_score(), 0.0)


class BetaStatTest(unittest.TestCase):
    def test_zero_data_returns_prior_mean(self):
        b = BetaStat(alpha=2, beta=5)
        self.assertAlmostEqual(b.estimate, 2 / 7, places=6)

    def test_estimate_converges_to_truth(self):
        rng = random.Random(0)
        b = BetaStat(alpha=2, beta=5)
        for _ in range(2000):
            b.observe(rng.random() < 0.6)
        self.assertAlmostEqual(b.estimate, 0.6, delta=0.03)

    def test_confidence_grows_with_data(self):
        b = BetaStat()
        c0 = b.confidence
        for _ in range(100):
            b.observe(True)
        self.assertGreater(b.confidence, c0)
        self.assertLess(b.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
