"""Headline demo: heads-up NLHE with the GTO preflop blueprint engaged.

Compares two bot configurations:
  Bot A: heuristic-only baseline + classifier deviations
  Bot B: GTO preflop blueprint (CFR-trained) + heuristic postflop + classifiers

Both face the same opponent profiles. We expect Bot B to outperform Bot A
because its preflop play is game-theoretically derived rather than hand-tuned.
"""
from __future__ import annotations

import random

from pokerbot.cache import (
    cached_real_classifiers,
    cached_preflop_hu_strategy,
)
from pokerbot.abstraction import cached_bucket_map
from pokerbot.games.nlhe import NLHEConfig
from pokerbot.runtime import (
    make_nlhe_player,
    play_table,
    MultiOpponentAdaptiveBot,
)


def main() -> None:
    print("Loading classifiers + bucket map + preflop CFR strategy...")
    bluff_clf, tilt_clf = cached_real_classifiers()
    bucket_map = cached_bucket_map()
    cfr_strategy = cached_preflop_hu_strategy(iterations=8000)

    print("\n=== Heads-up NLHE: bot vs each opponent profile ===")
    print(f"{'Opponent':<20} {'Bot A (heuristic)':>22} {'Bot B (GTO+heuristic)':>26} {'gain':>10}")
    print("-" * 82)

    n_hands = 1500
    for opp_name in ["maniac", "calling_station", "loose_aggressive",
                     "tight_passive", "tilt_prone"]:
        rng = random.Random(0)
        # Bot A: no CFR strategy
        bot_a = MultiOpponentAdaptiveBot(
            bluff_clf=bluff_clf, tilt_clf=tilt_clf,
            preflop_hu_strategy=None, bucket_map=None,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        opp_a = make_nlhe_player(opp_name, rng=random.Random(7))
        result_a = play_table([bot_a, opp_a], n_hands=n_hands,
                              config=NLHEConfig(n_players=2),
                              rng=random.Random(42))

        # Bot B: with CFR preflop blueprint
        rng = random.Random(0)
        bot_b = MultiOpponentAdaptiveBot(
            bluff_clf=bluff_clf, tilt_clf=tilt_clf,
            preflop_hu_strategy=cfr_strategy, bucket_map=bucket_map,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        opp_b = make_nlhe_player(opp_name, rng=random.Random(7))
        result_b = play_table([bot_b, opp_b], n_hands=n_hands,
                              config=NLHEConfig(n_players=2),
                              rng=random.Random(42))

        a_ev = result_a.seat_means[0]
        b_ev = result_b.seat_means[0]
        gain = b_ev - a_ev
        print(f"  vs {opp_name:<17} {a_ev:>+22.4f} {b_ev:>+26.4f} {gain:>+10.4f}  "
              f"({bot_b.n_preflop_cfr_lookups} CFR uses)")


if __name__ == "__main__":
    main()
