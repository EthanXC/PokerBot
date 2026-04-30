"""HEADLINE EXPERIMENT.

Compare:
  Bot A: pure CFR strategy (the GTO baseline)
  Bot B: AdaptiveBot — same CFR strategy + bluff/tilt classifiers

Both play the same opponents (each heuristic profile) for many hands.
Report mean EV with confidence intervals; the question is whether
Bot B (with the AI human-element layer) makes meaningfully more money
than Bot A against exploitable / tilting opponents.

This is the answer to 'does the AI catch-humans layer actually help.'
"""
from __future__ import annotations

import random
import time

from pokerbot.games.leduc import LeducPoker
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.learning import BluffClassifier, TiltClassifier
from pokerbot.runtime import (
    StrategyPlayer,
    AdaptiveBotPlayer,
    play_match,
    make_player,
    PROFILES,
    build_bluff_dataset,
    build_tilt_dataset,
)


def main() -> None:
    from pokerbot.cache import cached_cfr_leduc, cached_real_classifiers

    print("=== Setup ===")
    cfr_strategy = cached_cfr_leduc(iterations=300)
    bluff_clf, tilt_clf = cached_real_classifiers(n_hands_per_pairing=300, epochs=400)
    print()

    n_hands = 8000

    print(f"=== Match results — both bots play P0 vs each opponent profile, n={n_hands} ===")
    print(f"{'Opponent':<20} {'Pure CFR EV/hand':>22} {'Adaptive EV/hand':>22} {'Adaptive gain':>16}")
    print("-" * 84)

    rows = []
    for profile_name in PROFILES:
        # Pure CFR run
        cfr_p = StrategyPlayer(cfr_strategy, rng=random.Random(0))
        opp = make_player(profile_name, rng=random.Random(7))
        cfr_result = play_match(cfr_p, opp, n_hands, rng=random.Random(42))

        # Adaptive run
        adaptive = AdaptiveBotPlayer(
            cfr_strategy=cfr_strategy,
            bluff_clf=bluff_clf,
            tilt_clf=tilt_clf,
            deviation_strength=0.4,
            rng=random.Random(0),
        )
        opp2 = make_player(profile_name, rng=random.Random(7))
        adp_result = play_match(adaptive, opp2, n_hands, rng=random.Random(42))

        gain = adp_result.p0_mean - cfr_result.p0_mean
        rows.append((profile_name, cfr_result, adp_result, gain, adaptive.n_deviated, adaptive.n_decisions))

        cfr_str = f"{cfr_result.p0_mean:+.4f}"
        adp_str = f"{adp_result.p0_mean:+.4f}"
        print(f"{profile_name:<20} {cfr_str:>22} {adp_str:>22} {gain:>+16.4f}")

    print()
    print("Diagnostics — fraction of decisions where AdaptiveBot deviated from CFR:")
    for name, _, _, _, n_dev, n_dec in rows:
        rate = n_dev / max(1, n_dec)
        print(f"  vs {name:<18}: deviated {rate:6.1%} ({n_dev}/{n_dec} decisions)")

    print()
    print("Headline: AdaptiveBot's EV gain over pure CFR (positive = our AI helped):")
    avg_gain = sum(r[3] for r in rows) / len(rows)
    print(f"  Mean gain across {len(rows)} opponent profiles: {avg_gain:+.4f} chips/hand")
    print(f"  = {avg_gain * 1000 / 4:+.0f} mbb/g on average")


if __name__ == "__main__":
    main()
