"""Tests for the Dirichlet-MAP opponent strategy estimator."""
from __future__ import annotations

import random
import unittest

from pokerbot.learning.mle_strategy import StrategyMLE


class StrategyMLETest(unittest.TestCase):
    def test_no_data_returns_uniform(self):
        m = StrategyMLE(prior_alpha=1.0)
        m.observe_legal_actions("K:b", ["b", "p"])
        s = m.posterior_mean("K:b")
        self.assertAlmostEqual(s["b"], 0.5)
        self.assertAlmostEqual(s["p"], 0.5)

    def test_converges_to_truth_with_data(self):
        rng = random.Random(0)
        m = StrategyMLE(prior_alpha=1.0)
        m.observe_legal_actions("Q:b", ["b", "p"])
        # True opponent: bets 30% of the time at this info set.
        for _ in range(2000):
            a = "b" if rng.random() < 0.30 else "p"
            m.observe("Q:b", a)
        s = m.posterior_mean("Q:b")
        self.assertAlmostEqual(s["b"], 0.30, delta=0.03)
        self.assertAlmostEqual(s["p"], 0.70, delta=0.03)

    def test_strong_prior_resists_small_samples(self):
        m = StrategyMLE(prior_alpha=10.0)
        m.observe_legal_actions("J:p", ["b", "p"])
        # 3 observations all 'b' -- pure MLE would say p_bet = 1.0
        for _ in range(3):
            m.observe("J:p", "b")
        s = m.posterior_mean("J:p")
        # With prior_alpha=10, total mass = 10+10+3 = 23, b counts 10+3=13
        # Expected: 13/23 ~= 0.565, much less than 1.0
        self.assertLess(s["b"], 0.7)
        self.assertGreater(s["b"], 0.5)

    def test_pure_mle_no_shrinkage(self):
        m = StrategyMLE(prior_alpha=1.0)
        m.observe_legal_actions("J:p", ["b", "p"])
        for _ in range(3):
            m.observe("J:p", "b")
        # Pure MLE: 3/3 = 1.0
        s = m.mle("J:p")
        self.assertAlmostEqual(s["b"], 1.0)
        self.assertAlmostEqual(s["p"], 0.0)

    def test_confidence_grows_with_data(self):
        m = StrategyMLE(prior_alpha=2.0)
        m.observe_legal_actions("K:p", ["b", "p"])
        c0 = m.confidence("K:p")
        for _ in range(500):
            m.observe("K:p", "b")
        c1 = m.confidence("K:p")
        self.assertGreater(c1, c0)
        self.assertLess(c1, 1.0)

    def test_strategy_returns_full_dict(self):
        m = StrategyMLE()
        m.observe_legal_actions("A", ["x", "y"])
        m.observe_legal_actions("B", ["x", "y"])
        m.observe("A", "x")
        m.observe("A", "x")
        m.observe("B", "y")
        s = m.strategy()
        self.assertIn("A", s)
        self.assertIn("B", s)
        self.assertGreater(s["A"]["x"], s["A"]["y"])
        self.assertGreater(s["B"]["y"], s["B"]["x"])


if __name__ == "__main__":
    unittest.main()
