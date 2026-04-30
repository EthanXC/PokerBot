"""End-to-end demo: how the AI 'human-element' detection changes the bot's play.

Shows three scenarios on Kuhn poker:

  Scenario A — Calm opponent, balanced action: bluff_prob low, tilt low.
      Hybrid stays close to GTO. Safety preserved.

  Scenario B — Opponent has been losing AND just made a huge overbet on a
      weak board. Bluff classifier fires, tilt classifier fires.
      Hybrid pushes lambda up; perceived opponent range widens; bot
      bluff-catches lighter.

  Scenario C — Steady, balanced play but a big bet from a low-aggression
      villain on a strong board. Bluff classifier says LOW probability;
      bot stays tight, doesn't bluff-catch.

The point: the bot's strategy *changes* based on the bluff & tilt scores.
"""
from __future__ import annotations

import numpy as np

from pokerbot.games.kuhn import KuhnPoker
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.learning.bluff_classifier import BluffClassifier
from pokerbot.learning.tilt_classifier import TiltClassifier
from pokerbot.policy.human_aware import HumanAwarePolicy

from tests.test_bluff_classifier import synth_bluff_data
from tests.test_tilt_classifier import synth_tilt_data


def trained_gto(iters: int = 20_000) -> dict:
    g = KuhnPoker()
    s = CFRSolver(g, plus_regret_floor=True, linear_averaging=True)
    s.train(iters)
    return s.average_strategy()


def trained_bluff_clf(seed: int = 0) -> BluffClassifier:
    X, y = synth_bluff_data(2000, seed=seed)
    clf = BluffClassifier()
    clf.fit(X, y, epochs=400)
    return clf


def trained_tilt_clf(seed: int = 0) -> TiltClassifier:
    X, y = synth_tilt_data(2000, seed=seed)
    clf = TiltClassifier()
    clf.fit(X, y, epochs=400)
    return clf


def fmt_dist(d: dict) -> str:
    return ", ".join(f"{a}={p:.3f}" for a, p in sorted(d.items()))


def main() -> None:
    print("Training GTO blueprint, bluff classifier, tilt classifier...")
    gto = trained_gto(20_000)
    bluff_clf = trained_bluff_clf(0)
    tilt_clf = trained_tilt_clf(0)

    # The bot is P0 (in position). The opponent is P1.
    game = KuhnPoker()

    def make_bot() -> HumanAwarePolicy:
        return HumanAwarePolicy(
            game=game, gto_strategy=gto, player=0,
            bluff_clf=bluff_clf, tilt_clf=tilt_clf,
            base_lambda=0.6,
        )

    # Pretend we've watched the opponent play 200 hands and they've been
    # bluffing a lot (bets J:p often = bluffs). We feed observations.
    def feed_observations(bot: HumanAwarePolicy, p_bluffs_at_J_p: float = 0.6, n: int = 200) -> None:
        rng = np.random.default_rng(0)
        for _ in range(n):
            # Synthetic: opponent bets J after a check at rate p_bluffs_at_J_p
            bot.observe_action("J:p", "b" if rng.random() < p_bluffs_at_J_p else "p", ["b", "p"])
            # Other info sets stay near Nash for realism.
            bot.observe_action("Q:b", "b" if rng.random() < 0.5 else "p", ["b", "p"])
            bot.observe_action("K:b", "b", ["b", "p"])

    # ---- Scenario A: calm, balanced situation ----
    bot = make_bot()
    feed_observations(bot, p_bluffs_at_J_p=0.33, n=100)  # near-GTO opponent
    bluff_x = np.array([[0.6, 0.7, 0.4, 1.0, 0.4, 1, 0.0, 0.0]])  # value-bet feel
    tilt_x = np.array([[0.1, 0.0, 0.0, 15.0, 0.0, 1.0, 0.0]])     # not on tilt
    s_a = bot.make_decision(bet_info_sets=["J:p", "Q:p"], bluff_features=bluff_x, tilt_features=tilt_x)

    print("\n=== Scenario A — calm opponent, no tilt ===")
    print(f"  bluff_prob = {bot.last_bluff_prob:.3f}, tilt_prob = {bot.last_tilt_prob:.3f}")
    print(f"  effective lambda ~ {bot._hybrid.current_lambda:.3f}")
    print(f"  P0 strat at K:b (call K facing bet):  {fmt_dist(s_a.get('K:b', {}))}")
    print(f"  P0 strat at Q:pb (call Q facing bet): {fmt_dist(s_a.get('Q:pb', {}))}")

    # ---- Scenario B: opponent has been bluff-prone, BIG bet, signs of tilt ----
    bot = make_bot()
    feed_observations(bot, p_bluffs_at_J_p=0.65, n=200)  # very bluffy
    bluff_x = np.array([[1.4, 0.25, 0.8, 1.0, 0.6, 3, 1.0, 1.0]])  # huge bet, weak board
    tilt_x = np.array([[0.85, 0.18, 0.12, 1.0, 0.06, 5.0, 0.07]])  # all tilt signs
    s_b = bot.make_decision(bet_info_sets=["J:p", "Q:p"], bluff_features=bluff_x, tilt_features=tilt_x)

    print("\n=== Scenario B — bluff-heavy opponent, on tilt ===")
    print(f"  bluff_prob = {bot.last_bluff_prob:.3f}, tilt_prob = {bot.last_tilt_prob:.3f}")
    print(f"  effective lambda ~ {bot._hybrid.current_lambda:.3f}")
    print(f"  P0 strat at K:b (call K facing bet):  {fmt_dist(s_b.get('K:b', {}))}")
    print(f"  P0 strat at Q:pb (call Q facing bet): {fmt_dist(s_b.get('Q:pb', {}))}")
    print("  -> Bot should BLUFF-CATCH wider (Q:pb call rate up vs Scenario A).")

    # ---- Scenario C: tight opp, strong board, no tilt ----
    bot = make_bot()
    feed_observations(bot, p_bluffs_at_J_p=0.10, n=200)  # almost never bluffs
    bluff_x = np.array([[0.7, 0.85, 0.3, 0.0, 0.45, 1, 0.0, 0.0]])  # value-bet pattern
    tilt_x = np.array([[0.0, -0.02, 0.0, 25.0, 0.0, 0.0, 0.0]])     # rock-solid
    s_c = bot.make_decision(bet_info_sets=["J:p", "Q:p"], bluff_features=bluff_x, tilt_features=tilt_x)

    print("\n=== Scenario C — tight opponent, value-bet pattern, no tilt ===")
    print(f"  bluff_prob = {bot.last_bluff_prob:.3f}, tilt_prob = {bot.last_tilt_prob:.3f}")
    print(f"  effective lambda ~ {bot._hybrid.current_lambda:.3f}")
    print(f"  P0 strat at K:b (call K facing bet):  {fmt_dist(s_c.get('K:b', {}))}")
    print(f"  P0 strat at Q:pb (call Q facing bet): {fmt_dist(s_c.get('Q:pb', {}))}")
    print("  -> Bot should fold MORE marginal hands here.")

    # Summary of the swing
    print("\n=== Summary ===")
    print(f"  P0 calls Q:pb: scenario A={s_a['Q:pb'].get('b', 0):.3f},"
          f" B={s_b['Q:pb'].get('b', 0):.3f}, C={s_c['Q:pb'].get('b', 0):.3f}")
    print("  Calling rate at Q:pb should be highest in Scenario B (against tilted")
    print("  bluffer) and lowest in Scenario C (against tight value-better).")


if __name__ == "__main__":
    main()
