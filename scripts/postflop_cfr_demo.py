"""Demo: heads-up NLHE with naive preflop chart + postflop CFR + classifiers.

Compares:
  Bot A: pure heuristic baseline + classifiers (no game theory)
  Bot B: chart preflop + postflop CFR + classifiers (game theory engaged
         where it matters most: postflop)
"""
from __future__ import annotations

import random

from pokerbot.cache import (
    cached_real_classifiers,
    cached_postflop_hu_strategy,
    cached_nlhe_bluff_classifier,
)
from pokerbot.games.nlhe import NLHEConfig
from pokerbot.runtime import (
    make_nlhe_player,
    play_table,
    MultiOpponentAdaptiveBot,
)


def main() -> None:
    print("Loading classifiers + postflop CFR strategy...")
    # Use the NLHE-trained bluff classifier (better domain match than Leduc one).
    bluff_clf = cached_nlhe_bluff_classifier()
    _, tilt_clf = cached_real_classifiers()
    postflop_strategy = cached_postflop_hu_strategy()

    n_hands = 1500
    print(f"\n=== Heads-up NLHE: bot vs each opponent profile ({n_hands} hands) ===")
    print(f"{'Opponent':<22} {'A: heuristic':>15} {'B: chart+postflopCFR':>22} "
          f"{'gain':>10} {'CFR uses':>10}")
    print("-" * 82)

    for opp_name in ["maniac", "calling_station", "loose_aggressive",
                     "tight_passive", "tilt_prone"]:
        # Bot A: pure heuristic (no chart, no postflop CFR)
        rng = random.Random(0)
        bot_a = MultiOpponentAdaptiveBot(
            bluff_clf=bluff_clf, tilt_clf=tilt_clf,
            use_preflop_chart=False,
            postflop_hu_strategy=None,
            bluff_threshold=0.50,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        opp_a = make_nlhe_player(opp_name, rng=random.Random(7))
        result_a = play_table([bot_a, opp_a], n_hands=n_hands,
                              config=NLHEConfig(n_players=2),
                              rng=random.Random(42))

        # Bot B: chart preflop + postflop CFR + classifier-driven exploit
        rng = random.Random(0)
        bot_b = MultiOpponentAdaptiveBot(
            bluff_clf=bluff_clf, tilt_clf=tilt_clf,
            use_preflop_chart=True,
            postflop_hu_strategy=postflop_strategy,
            bluff_threshold=0.50,
            deviation_strength=0.50,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        opp_b = make_nlhe_player(opp_name, rng=random.Random(7))
        result_b = play_table([bot_b, opp_b], n_hands=n_hands,
                              config=NLHEConfig(n_players=2),
                              rng=random.Random(42))

        a_ev = result_a.seat_means[0]
        b_ev = result_b.seat_means[0]
        gain = b_ev - a_ev
        print(f"  vs {opp_name:<19} {a_ev:>+15.4f} {b_ev:>+22.4f} "
              f"{gain:>+10.4f} {bot_b.n_postflop_cfr_lookups:>10}")
        print(f"  {'  (B diagnostics)':<22} "
              f"chart={bot_b.n_preflop_chart_lookups}, "
              f"postflopCFR={bot_b.n_postflop_cfr_lookups}, "
              f"deviated={bot_b.n_deviated}")


if __name__ == "__main__":
    main()
