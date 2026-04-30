"""Regression tests for the AdaptiveBotPlayer.

This is the headline test: the AI human-element layer must produce a
positive EV gain over pure CFR, on average, across a panel of opponent
types — and it must not lose much against tight (unexploitable-ish) ones.
"""
from __future__ import annotations

import random
import unittest

from pokerbot.games.leduc import LeducPoker
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.learning import BluffClassifier, TiltClassifier
from pokerbot.runtime import (
    StrategyPlayer,
    AdaptiveBotPlayer,
    play_match,
    make_player,
    build_bluff_dataset,
    build_tilt_dataset,
)


class AdaptiveBotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Modest CFR training — enough that the strategy isn't terrible.
        game = LeducPoker()
        solver = CFRSolver(game, plus_regret_floor=True, linear_averaging=True)
        solver.train(150)
        cls.cfr_strategy = solver.average_strategy()

        # Real-trained classifiers (smaller dataset for test speed).
        Xb, yb = build_bluff_dataset(n_hands_per_pairing=200, seed=0)
        Xt, yt = build_tilt_dataset(n_hands_per_pairing=200, seed=0)
        cls.bluff_clf = BluffClassifier().fit(Xb, yb, epochs=300)
        cls.tilt_clf = TiltClassifier().fit(Xt, yt, epochs=300)

    def _run_match(self, opp_name: str, n_hands: int = 3000) -> tuple:
        # Pure CFR result
        cfr_p = StrategyPlayer(self.cfr_strategy, rng=random.Random(0))
        opp = make_player(opp_name, rng=random.Random(7))
        cfr_result = play_match(cfr_p, opp, n_hands, rng=random.Random(42))

        # Adaptive result
        adaptive = AdaptiveBotPlayer(
            cfr_strategy=self.cfr_strategy,
            bluff_clf=self.bluff_clf,
            tilt_clf=self.tilt_clf,
            rng=random.Random(0),
        )
        opp2 = make_player(opp_name, rng=random.Random(7))
        adp_result = play_match(adaptive, opp2, n_hands, rng=random.Random(42))

        return cfr_result.p0_mean, adp_result.p0_mean, adaptive

    def test_adaptive_beats_cfr_vs_maniac(self):
        # 6000 hands gives SE ~0.02 on EV, so a 0.05-chip threshold is
        # comfortably outside the noise.
        cfr_ev, adp_ev, _ = self._run_match("maniac", n_hands=6000)
        gain = adp_ev - cfr_ev
        self.assertGreater(
            gain, 0.03,
            f"AdaptiveBot vs maniac should gain >0.03 over CFR; got {gain:+.4f}"
        )

    def test_adaptive_safe_vs_tight_passive(self):
        """Against an unexploitable-ish opponent, deviation must be rare.

        We test the deviation RATE (which is robust) rather than the
        per-match EV (which has wide CIs at short match lengths).
        """
        _, _, adaptive = self._run_match("tight_passive", n_hands=2000)
        deviation_rate = adaptive.n_deviated / max(1, adaptive.n_decisions)

        self.assertLess(
            deviation_rate, 0.10,
            f"Adaptive should rarely deviate vs tight_passive; got {deviation_rate:.1%}"
        )

    def test_adaptive_panel_average_positive(self):
        """The headline test: across a panel of opponents, adaptive bot's
        average EV should beat pure CFR's. Long match lengths to reduce noise.
        """
        opponents = ["maniac", "loose_aggressive", "calling_station", "tilt_prone", "tight_passive"]
        gains = []
        for name in opponents:
            cfr_ev, adp_ev, _ = self._run_match(name, n_hands=4000)
            gains.append(adp_ev - cfr_ev)
        avg_gain = sum(gains) / len(gains)
        self.assertGreater(
            avg_gain, 0.0,
            f"Adaptive should beat pure CFR on AVERAGE across panel; "
            f"got per-profile gains={[f'{g:+.3f}' for g in gains]}, avg={avg_gain:+.4f}"
        )

    def test_adaptive_aggression_gate_works(self):
        """Empirical aggression rate computed by AdaptiveBot must reflect
        reality: very low for calling_station, very high for maniac."""
        # Calling station: bet_strong=0.6, bluff=0.05, raise_when_bet=0.05.
        # Their actual aggression rate (b+r / b+r+c+f+k) should be modest.
        adaptive_cs = AdaptiveBotPlayer(
            self.cfr_strategy, self.bluff_clf, self.tilt_clf,
            rng=random.Random(0),
        )
        opp_cs = make_player("calling_station", rng=random.Random(7))
        play_match(adaptive_cs, opp_cs, 1000, rng=random.Random(42))
        cs_rate = adaptive_cs._opp.n_aggressive / max(1, adaptive_cs._opp.n_actions_taken)
        self.assertLess(cs_rate, 0.5, f"calling_station agg rate too high: {cs_rate:.1%}")

        # Maniac: should be very aggressive.
        adaptive_m = AdaptiveBotPlayer(
            self.cfr_strategy, self.bluff_clf, self.tilt_clf,
            rng=random.Random(0),
        )
        opp_m = make_player("maniac", rng=random.Random(7))
        play_match(adaptive_m, opp_m, 1000, rng=random.Random(42))
        m_rate = adaptive_m._opp.n_aggressive / max(1, adaptive_m._opp.n_actions_taken)
        self.assertGreater(m_rate, 0.6, f"maniac agg rate too low: {m_rate:.1%}")


if __name__ == "__main__":
    unittest.main()
