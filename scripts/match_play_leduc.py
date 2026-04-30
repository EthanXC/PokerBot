"""Bot-vs-bot match-play on Leduc with confidence intervals.

Trains a CFR strategy on Leduc, then plays it against each heuristic
opponent profile. Reports mean chips/hand and 95% CI.

Expected:
  - vs calling_station, maniac: positive EV (CFR exploits them)
  - vs tight_passive, loose_aggressive: positive but smaller
  - vs another CFR strategy: ~0 (both near Nash)
"""
from __future__ import annotations

import random
import time

from pokerbot.games.leduc import LeducPoker
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.runtime import StrategyPlayer, play_match, make_player, PROFILES


def main() -> None:
    print("Training CFR+ on Leduc (300 iters, ~30s)...")
    t0 = time.time()
    game = LeducPoker()
    solver = CFRSolver(game, plus_regret_floor=True, linear_averaging=True)
    solver.train(300)
    cfr_strategy = solver.average_strategy()
    print(f"  done in {time.time()-t0:.1f}s\n")

    n_hands = 5000

    print(f"Match-play results (CFR strategy as P0, vs each heuristic profile as P1)")
    print(f"  n_hands = {n_hands} per matchup; CI = 95%; chips/hand")
    print("-" * 75)

    for profile_name in PROFILES:
        cfr_p = StrategyPlayer(cfr_strategy, rng=random.Random(0))
        heur_p = make_player(profile_name, rng=random.Random(7))
        result = play_match(cfr_p, heur_p, n_hands, rng=random.Random(42))
        positive = "+" if result.p0_mean > 0 else " "
        print(f"  vs {profile_name:<18} {positive}{result.p0_mean:+.4f}  "
              f"CI [{result.p0_ci_low:+.4f}, {result.p0_ci_high:+.4f}]  "
              f"= {result.mbb_per_game:+.0f} mbb/g")

    # Also play CFR vs CFR — should be roughly zero.
    cfr_a = StrategyPlayer(cfr_strategy, rng=random.Random(99))
    cfr_b = StrategyPlayer(cfr_strategy, rng=random.Random(100))
    result = play_match(cfr_a, cfr_b, n_hands, rng=random.Random(43))
    print(f"  vs CFR              {result.p0_mean:+.4f}  "
          f"CI [{result.p0_ci_low:+.4f}, {result.p0_ci_high:+.4f}]  "
          f"= {result.mbb_per_game:+.0f} mbb/g    (should straddle 0)")


if __name__ == "__main__":
    main()
