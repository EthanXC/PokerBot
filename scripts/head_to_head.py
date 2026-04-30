"""Head-to-head EV comparison: GTO vs. various opponent types,
and hybrid-bot vs. various opponent types.

Outputs a table that makes the value of the hybrid layer obvious:

                        vs. NASH      vs. CALLING STATION    vs. NIT
    Pure GTO            -0.0556        -0.0556                 ...
    Hybrid (lam=1)      -0.0556        much better             much better

Run:
    python -m scripts.head_to_head
"""
from __future__ import annotations

from pokerbot.games.kuhn import KuhnPoker
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.policy.hybrid import HybridPolicy
from tests.test_cfr_convergence import expected_value_p0
from tests.test_exploitability import kuhn_nash_strategy
from tests.test_hybrid import always_call_p1


def trained_gto(iters: int = 20_000) -> dict:
    g = KuhnPoker()
    s = CFRSolver(g, plus_regret_floor=True, linear_averaging=True)
    s.train(iters)
    return s.average_strategy()


def all_fold_p1() -> dict:
    """Nit on steroids: folds to anything."""
    return {
        "J:p": {"b": 0.0, "p": 1.0},
        "Q:p": {"b": 0.0, "p": 1.0},
        "K:p": {"b": 0.0, "p": 1.0},
        "J:b": {"b": 0.0, "p": 1.0},
        "Q:b": {"b": 0.0, "p": 1.0},
        "K:b": {"b": 0.0, "p": 1.0},
    }


def value_for_p0(p0_strategy: dict, p1_strategy: dict) -> float:
    g = KuhnPoker()
    combined = dict(p0_strategy)
    combined.update(p1_strategy)
    return expected_value_p0(g, combined)


def main() -> None:
    print("Training GTO blueprint on Kuhn (20k iters)...")
    gto = trained_gto(20_000)

    # Restrict gto to P0's info sets only for our experiments.
    p0_keys = {"J:", "Q:", "K:", "J:pb", "Q:pb", "K:pb"}
    gto_p0 = {k: v for k, v in gto.items() if k in p0_keys}
    gto_p1 = {k: v for k, v in gto.items() if k not in p0_keys}

    opponents = {
        "Nash":             kuhn_nash_strategy(alpha=1 / 6),
        "calling-station":  always_call_p1(),
        "all-fold-nit":     all_fold_p1(),
        "GTO-trained":      {k: v for k, v in gto.items() if k not in p0_keys},
    }

    # Subset opponent strategies to P1's info sets.
    p1_keys = {"J:p", "Q:p", "K:p", "J:b", "Q:b", "K:b"}
    opponents = {name: {k: v for k, v in s.items() if k in p1_keys} for name, s in opponents.items()}

    game = KuhnPoker()

    print("\nEV TO P0 (chips per hand). Higher = better for P0.")
    print(f"{'OPPONENT':<20} {'Pure GTO':>12} {'Hybrid lam=1':>15} {'gain':>10}")
    print("-" * 60)

    for name, opp in opponents.items():
        ev_gto = value_for_p0(gto_p0, opp)

        hp = HybridPolicy(game, gto, player=0, base_lambda=1.0)
        hp.update_opponent_model(opp, confidence=1.0, tilt=0.0)
        mixed = hp.strategy()
        mixed_p0 = {k: v for k, v in mixed.items() if k in p0_keys}
        ev_hyb = value_for_p0(mixed_p0, opp)

        delta = ev_hyb - ev_gto
        print(f"{name:<20} {ev_gto:>+12.4f} {ev_hyb:>+15.4f} {delta:>+10.4f}")

    print("\nReadout:")
    print("  - vs. Nash: hybrid should equal GTO (it's already unbeatable).")
    print("  - vs. calling-station / all-fold-nit: hybrid wins MUCH more.")
    print("  - vs. GTO-trained P1: gap is small (also near-Nash).")


if __name__ == "__main__":
    main()
