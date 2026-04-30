"""Headline demo: 6-handed NLHE simulation with the AI bot at the table.

Setup:
  Seat 0: MultiOpponentAdaptiveBot (the bot, using bluff/tilt classifiers
                                    trained on Leduc traces — yes, transfer
                                    learning across game sizes!)
  Seats 1-5: Mix of heuristic profiles
"""
from __future__ import annotations

import random

from pokerbot.cache import cached_real_classifiers, cached_nlhe_bluff_classifier
from pokerbot.games.nlhe import NLHEConfig
from pokerbot.runtime import (
    make_nlhe_player,
    play_table,
    MultiOpponentAdaptiveBot,
)


def main() -> None:
    print("Loading classifiers...")
    # NLHE-specific bluff classifier (trained on NLHE simulator traces)
    bluff_clf = cached_nlhe_bluff_classifier(n_hands=2000)
    # Tilt classifier transfers fine — its features are game-agnostic
    _, tilt_clf = cached_real_classifiers()

    print("\n=== 6-handed NLHE: AI bot vs 5 heuristic opponents ===")
    rng = random.Random(0)

    # The bot at seat 0; mixed heuristics at 1-5.
    # Conservative tuning: only deviate when highly confident; small flip prob.
    bot = MultiOpponentAdaptiveBot(
        bluff_clf=bluff_clf, tilt_clf=tilt_clf,
        baseline_profile="tight_passive",
        deviation_strength=0.20,
        bluff_threshold=0.75,
        tilt_threshold=0.65,
        rng=random.Random(rng.randint(0, 2**31)),
    )
    opp_names = ["loose_aggressive", "calling_station", "maniac", "tilt_prone", "tight_passive"]
    opps = [make_nlhe_player(n, rng=random.Random(rng.randint(0, 2**31))) for n in opp_names]

    players = [bot] + opps
    n_hands = 3000
    print(f"Playing {n_hands} hands at a 6-handed table...")
    result = play_table(players, n_hands=n_hands, rng=random.Random(42))

    print(f"\n  seat 0 (BOT, baseline=tight_passive)            "
          f"{result.seat_means[0]:+.4f} chips/hand "
          f"CI [{result.seat_ci_low[0]:+.4f}, {result.seat_ci_high[0]:+.4f}]")
    for i, name in enumerate(opp_names, start=1):
        print(f"  seat {i} ({name:<24})  "
              f"{result.seat_means[i]:+.4f} chips/hand "
              f"CI [{result.seat_ci_low[i]:+.4f}, {result.seat_ci_high[i]:+.4f}]")

    print(f"\nBot diagnostics:")
    print(f"  decisions made:       {bot.n_decisions}")
    print(f"  classifier deviations: {bot.n_deviated} ({bot.n_deviated/max(1, bot.n_decisions):.1%})")

    # Per-opponent state
    print(f"\nPer-opponent stats (as the bot saw them):")
    print(f"  {'seat':<6} {'hands':>6} {'actions':>9} {'agg%':>6} {'showdown%':>11}")
    for seat in sorted(bot._stats.keys()):
        s = bot._stats[seat]
        if s.n_actions_taken == 0:
            continue
        agg_rate = s.n_aggressive / s.n_actions_taken
        sd_rate = s.n_showdowns / max(1, s.n_hands)
        print(f"  {seat:<6} {s.n_hands:>6} {s.n_actions_taken:>9} "
              f"{agg_rate:>5.1%} {sd_rate:>10.1%}")

    print(f"\nTotal at table: {sum(result.seat_means):+.4f} (sanity check, should be ~0)")


if __name__ == "__main__":
    main()
