"""Train CFR on Kuhn poker and print the strategy table next to the
analytical Nash equilibrium so you can eyeball convergence.

    python -m scripts.train_kuhn [iterations] [--vanilla]
"""
from __future__ import annotations

import sys

from pokerbot.games.kuhn import KuhnPoker
from pokerbot.solvers.cfr import CFRSolver
from tests.test_cfr_convergence import expected_value_p0


def fmt_strategy(strat: dict[str, float]) -> str:
    return ", ".join(f"{a}={p:.3f}" for a, p in sorted(strat.items()))


def main() -> None:
    iters = 20_000
    plus = True
    if len(sys.argv) > 1:
        try:
            iters = int(sys.argv[1])
        except ValueError:
            pass
    if "--vanilla" in sys.argv:
        plus = False

    game = KuhnPoker()
    solver = CFRSolver(game, plus_regret_floor=plus, linear_averaging=plus)
    print(f"Training {'CFR+' if plus else 'vanilla CFR'} for {iters} iterations on Kuhn...")
    solver.train(iters, verbose_every=max(iters // 5, 1))

    avg = solver.average_strategy()
    v = expected_value_p0(game, avg)
    print(f"\nGame value to P0: {v:+.5f}    (Nash: {-1/18:+.5f})")
    print(f"Exploit gap (rough): {abs(v - (-1/18)):.5f}\n")

    print("Player 0 strategy (first to act):")
    for key in ["J:", "Q:", "K:", "J:pb", "Q:pb", "K:pb"]:
        if key in avg:
            print(f"  {key:<6}  {fmt_strategy(avg[key])}")

    print("\nPlayer 1 strategy (response):")
    for key in ["J:p", "Q:p", "K:p", "J:b", "Q:b", "K:b"]:
        if key in avg:
            print(f"  {key:<6}  {fmt_strategy(avg[key])}")

    # Highlight the headline numbers vs. analytical Nash.
    print("\n--- Comparison vs. Nash ---")
    print(f"  P0 bet K  / 3*bet J  : {avg['K:']['b']:.3f} vs {3*avg['J:']['b']:.3f}   (should match)")
    print(f"  P1 call Q-bet         : {avg['Q:b']['b']:.3f}    (Nash: 0.333)")
    print(f"  P1 bluff J on check   : {avg['J:p']['b']:.3f}    (Nash: 0.333)")
    print(f"  P1 fold J facing bet  : {avg['J:b']['p']:.3f}    (Nash: 1.000)")
    print(f"  P1 call K facing bet  : {avg['K:b']['b']:.3f}    (Nash: 1.000)")


if __name__ == "__main__":
    main()
