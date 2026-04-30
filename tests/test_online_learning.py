"""Test that AdaptiveBotPlayer's online classifier updates work and
make predictions move in the right direction.
"""
from __future__ import annotations

import random
import unittest

import numpy as np

from pokerbot.cache import cached_cfr_leduc, cached_real_classifiers
from pokerbot.runtime import (
    AdaptiveBotPlayer,
    play_match,
    make_player,
)
from pokerbot.learning import BluffClassifier
from tests.test_bluff_classifier import synth_bluff_data


class PartialFitTest(unittest.TestCase):
    def test_partial_fit_moves_predictions(self):
        """One online step should move the prediction toward the label."""
        X, y = synth_bluff_data(500, seed=0)
        clf = BluffClassifier().fit(X, y, epochs=200)
        # Pick an example where the model gets it wrong (or at least not extreme).
        p_before = float(clf.predict_proba(X[0:1])[0])

        # Several online updates with the SAME (x, y) should drive the
        # prediction toward y.
        for _ in range(50):
            clf.partial_fit(X[0:1], y[0:1], lr=0.1)
        p_after = float(clf.predict_proba(X[0:1])[0])
        if y[0] == 1:
            self.assertGreater(p_after, p_before, "should have moved toward 1")
        else:
            self.assertLess(p_after, p_before, "should have moved toward 0")

    def test_partial_fit_requires_fit_first(self):
        clf = BluffClassifier()
        with self.assertRaises(RuntimeError):
            clf.partial_fit(np.zeros((1, 12)), np.array([0]))


class OnlineLearningInMatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfr = cached_cfr_leduc(iterations=300)
        cls.bluff_clf, cls.tilt_clf = cached_real_classifiers(
            n_hands_per_pairing=300, epochs=400
        )

    def test_observe_showdown_does_work(self):
        """Run a match with a known-bluffy opponent and verify the bot's
        online_updates counter increased.
        """
        # Use a fresh BluffClassifier so we don't mutate the cached one.
        from copy import deepcopy
        bluff_clf = deepcopy(self.bluff_clf)
        tilt_clf = deepcopy(self.tilt_clf)

        bot = AdaptiveBotPlayer(
            cfr_strategy=self.cfr,
            bluff_clf=bluff_clf,
            tilt_clf=tilt_clf,
            rng=random.Random(0),
        )
        opp = make_player("maniac", rng=random.Random(7))
        play_match(bot, opp, 1000, rng=random.Random(42))

        # Some online updates should have happened.
        self.assertGreater(bot.n_online_updates, 50,
                           f"got only {bot.n_online_updates} online updates")

    def test_online_learning_does_not_break_match_play(self):
        """Sanity: matches still complete and produce reasonable EV with
        online updates active."""
        from copy import deepcopy
        bluff_clf = deepcopy(self.bluff_clf)
        tilt_clf = deepcopy(self.tilt_clf)

        bot = AdaptiveBotPlayer(
            cfr_strategy=self.cfr,
            bluff_clf=bluff_clf,
            tilt_clf=tilt_clf,
            rng=random.Random(0),
        )
        opp = make_player("loose_aggressive", rng=random.Random(7))
        result = play_match(bot, opp, 800, rng=random.Random(42))
        # Should still beat loose_aggressive (it's exploitable).
        self.assertGreater(result.p0_mean, 0.0,
                           f"online-updating bot should still beat loose_aggressive; "
                           f"got {result.p0_mean:+.4f}")


if __name__ == "__main__":
    unittest.main()
