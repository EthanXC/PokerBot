"""Tests for the hybrid GTO + exploit policy.

This is the headline correctness check for the bot's whole architecture:
the hybrid policy must beat pure GTO when facing a known-exploitable
opponent, AND must not lose to GTO when facing a balanced (Nash) opponent.

Setup: use Kuhn poker (where we know Nash). Build a deliberately bad
opponent strategy (always-call), then verify:
  1. mix_strategies(GTO, BR, 0)   == GTO
  2. mix_strategies(GTO, BR, 1)   == BR (deterministic-best-response)
  3. EV against the bad opponent: hybrid (lam=1) > pure GTO (lam=0)
  4. EV against Nash opponent:    hybrid (lam=positive) does not lose
     significantly more than pure GTO (which is the safety guarantee).
"""
from __future__ import annotations

import unittest

from pokerbot.games.kuhn import KuhnPoker
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.policy.hybrid import HybridPolicy, mix_strategies, best_response_strategy
from tests.test_cfr_convergence import expected_value_p0


def trained_kuhn_strategy(iters: int = 10_000) -> dict:
    game = KuhnPoker()
    solver = CFRSolver(game, plus_regret_floor=True, linear_averaging=True)
    solver.train(iters)
    return solver.average_strategy()


def always_call_p1() -> dict:
    """A weak P1: never folds, never bluffs. Should be exploited by always
    value-betting strong hands and never bluffing junk."""
    return {
        "J:p":  {"b": 0.0, "p": 1.0},
        "Q:p":  {"b": 0.0, "p": 1.0},
        "K:p":  {"b": 1.0, "p": 0.0},
        "J:b":  {"b": 1.0, "p": 0.0},  # never folds (calls J facing bet — terrible)
        "Q:b":  {"b": 1.0, "p": 0.0},  # never folds
        "K:b":  {"b": 1.0, "p": 0.0},  # always calls
    }


class HybridMixingTest(unittest.TestCase):
    def test_lambda_zero_returns_gto(self):
        gto = trained_kuhn_strategy()
        opp_p1 = always_call_p1()
        hp = HybridPolicy(KuhnPoker(), gto, player=0, base_lambda=0.5)
        hp.update_opponent_model(opp_p1, confidence=0.0)  # confidence=0 -> lam=0
        # With lam = 0, the strategy must equal gto.
        self.assertEqual(hp.current_lambda, 0.0)
        for k, dist in gto.items():
            actual = hp.strategy().get(k, {})
            for a, p in dist.items():
                self.assertAlmostEqual(actual.get(a, 0.0), p, places=9)

    def test_lambda_one_returns_br(self):
        gto = trained_kuhn_strategy()
        opp_p1 = always_call_p1()
        hp = HybridPolicy(KuhnPoker(), gto, player=0, base_lambda=1.0)
        hp.update_opponent_model(opp_p1, confidence=1.0, tilt=0.0)
        self.assertAlmostEqual(hp.current_lambda, 1.0)
        # And the strategy at the info sets BR knows about should be deterministic.
        for k, dist in hp.exploit_br.items():
            mixed = hp.strategy()[k]
            for a, p in dist.items():
                self.assertAlmostEqual(mixed.get(a, 0.0), p, places=9)

    def test_mix_distribution_sums_to_one(self):
        gto = trained_kuhn_strategy()
        opp_p1 = always_call_p1()
        hp = HybridPolicy(KuhnPoker(), gto, player=0, base_lambda=0.6)
        hp.update_opponent_model(opp_p1, confidence=1.0)
        for k, dist in hp.strategy().items():
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=6, msg=f"key={k}")


class HybridExploitsWeakOpponentTest(unittest.TestCase):
    """The headline test."""

    def test_hybrid_beats_pure_gto_vs_calling_station(self):
        game = KuhnPoker()
        gto = trained_kuhn_strategy()
        opp_p1 = always_call_p1()

        # Pure GTO: combine GTO P0 with the calling-station P1.
        pure = dict(gto)
        for k, dist in opp_p1.items():
            pure[k] = dist
        ev_pure = expected_value_p0(game, pure)

        # Hybrid (lam=1): use full BR vs. opponent.
        hp = HybridPolicy(game, gto, player=0, base_lambda=1.0)
        hp.update_opponent_model(opp_p1, confidence=1.0, tilt=0.0)
        mixed = dict(hp.strategy())
        for k, dist in opp_p1.items():
            mixed[k] = dist
        ev_hybrid = expected_value_p0(game, mixed)

        # Hybrid must yield strictly more than pure GTO.
        self.assertGreater(
            ev_hybrid, ev_pure + 0.01,
            f"hybrid should outperform GTO vs calling station: "
            f"ev_pure={ev_pure:+.4f}, ev_hybrid={ev_hybrid:+.4f}"
        )

    def test_safety_vs_balanced_opponent(self):
        """Against a Nash-equilibrium P1, hybrid (with low lambda) shouldn't
        bleed significant EV vs. pure GTO."""
        game = KuhnPoker()
        gto = trained_kuhn_strategy()

        # P1 plays an EXACT Nash strategy.
        from tests.test_exploitability import kuhn_nash_strategy
        nash_p1_subset = {
            k: v for k, v in kuhn_nash_strategy(alpha=1 / 6).items() if k.endswith(":p") or k.endswith(":b")
        }

        pure = dict(gto)
        pure.update(nash_p1_subset)
        ev_pure = expected_value_p0(game, pure)

        # Use confidence=0.3 (modest data) — lam ends up modest.
        hp = HybridPolicy(game, gto, player=0, base_lambda=0.5)
        hp.update_opponent_model(nash_p1_subset, confidence=0.3, tilt=0.0)
        mixed = dict(hp.strategy())
        mixed.update(nash_p1_subset)
        ev_hybrid = expected_value_p0(game, mixed)

        # Against true Nash, BR can't do better than -1/18 either, but it
        # might do worse if lam=1 fixes us to a deterministic strategy that
        # pushes us against a Nash range. With moderate lam, the gap should
        # be small — say < 0.02.
        self.assertGreaterEqual(
            ev_hybrid, ev_pure - 0.02,
            f"hybrid should not catastrophically lose vs Nash: "
            f"ev_pure={ev_pure:+.4f}, ev_hybrid={ev_hybrid:+.4f}"
        )


if __name__ == "__main__":
    unittest.main()
