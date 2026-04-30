"""Train CFR on Leduc poker and watch exploitability decrease.

Vanilla CFR on Leduc is slow but tractable — the game has ~O(10^4) states.
We log exploitability every N iterations so you can see the curve.

    python -m scripts.train_leduc [iterations]
"""
from __future__ import annotations

import sys
import time

from pokerbot.games.leduc import LeducPoker
from pokerbot.solvers.cfr import CFRSolver
from pokerbot.solvers.exploitability import exploitability


def main() -> None:
    iters = 500
    if len(sys.argv) > 1:
        try:
            iters = int(sys.argv[1])
        except ValueError:
            pass

    game = LeducPoker()
    solver = CFRSolver(game, plus_regret_floor=True, linear_averaging=True)

    print(f"Training CFR+ on Leduc for {iters} iterations.")
    print(f"{'iter':>6} | {'time(s)':>8} | {'expl(mbb/g)':>12} | {'info_sets':>9}")
    print("-" * 50)

    log_every = max(iters // 10, 1)
    t0 = time.time()
    for t in range(1, iters + 1):
        for player in range(game.NUM_PLAYERS):
            solver.iteration = t
            solver._cfr(game.initial_state(), player, [1.0, 1.0])
        if t % log_every == 0 or t == iters:
            avg = solver.average_strategy()
            e = exploitability(game, avg)
            # Convert to mbb/game (milli-big-blinds per game). For Leduc, big bet
            # is 4 chips so 1 chip ~= 250 mbb. Just for human-friendly scale.
            mbb = e * 1000.0 / 4.0
            elapsed = time.time() - t0
            print(f"{t:>6} | {elapsed:>8.2f} | {mbb:>12.2f} | {len(solver.tables):>9}")


if __name__ == "__main__":
    main()
