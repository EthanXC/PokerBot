"""Regression tests for the NLHE match harness + heuristic players."""
from __future__ import annotations

import random
import unittest

from pokerbot.games.nlhe import NLHEConfig
from pokerbot.runtime import make_nlhe_player, play_table, NLHE_PROFILES


class TableMatchTest(unittest.TestCase):
    def test_six_handed_zero_sum(self):
        rng = random.Random(0)
        players = [
            make_nlhe_player("tight_passive", rng=random.Random(rng.randint(0, 2**31))),
            make_nlhe_player("loose_aggressive", rng=random.Random(rng.randint(0, 2**31))),
            make_nlhe_player("calling_station", rng=random.Random(rng.randint(0, 2**31))),
            make_nlhe_player("maniac", rng=random.Random(rng.randint(0, 2**31))),
            make_nlhe_player("tilt_prone", rng=random.Random(rng.randint(0, 2**31))),
            make_nlhe_player("tight_passive", rng=random.Random(rng.randint(0, 2**31))),
        ]
        result = play_table(players, n_hands=100, rng=random.Random(42))
        # Zero-sum across the table.
        self.assertAlmostEqual(sum(result.seat_means), 0.0, places=6)

    def test_nine_handed_runs(self):
        rng = random.Random(0)
        # 9-handed mix of profiles
        names = ["tight_passive", "loose_aggressive", "calling_station",
                 "maniac", "tilt_prone", "tight_passive", "loose_aggressive",
                 "calling_station", "tilt_prone"]
        players = [make_nlhe_player(n, rng=random.Random(rng.randint(0, 2**31)))
                   for n in names]
        result = play_table(
            players, n_hands=50,
            config=NLHEConfig(n_players=9),
            rng=random.Random(42),
        )
        self.assertEqual(len(result.seat_means), 9)
        self.assertAlmostEqual(sum(result.seat_means), 0.0, places=6)

    def test_ten_handed_runs(self):
        """Smoke test the maximum supported table size."""
        rng = random.Random(0)
        players = [make_nlhe_player("tight_passive", rng=random.Random(rng.randint(0, 2**31)))
                   for _ in range(10)]
        result = play_table(
            players, n_hands=20,
            config=NLHEConfig(n_players=10),
            rng=random.Random(42),
        )
        self.assertEqual(len(result.seat_means), 10)
        self.assertAlmostEqual(sum(result.seat_means), 0.0, places=6)

    def test_tight_beats_maniac_at_6max(self):
        """Three tight players + three maniacs. Tight should win on average."""
        rng = random.Random(0)
        names = ["tight_passive", "maniac", "tight_passive",
                 "maniac", "tight_passive", "maniac"]
        players = [make_nlhe_player(n, rng=random.Random(rng.randint(0, 2**31)))
                   for n in names]
        result = play_table(players, n_hands=400, rng=random.Random(42))
        tight_total = result.seat_means[0] + result.seat_means[2] + result.seat_means[4]
        maniac_total = result.seat_means[1] + result.seat_means[3] + result.seat_means[5]
        self.assertGreater(tight_total, maniac_total,
                           f"tight should beat maniac at 6max; got tight={tight_total}, "
                           f"maniac={maniac_total}")


if __name__ == "__main__":
    unittest.main()
