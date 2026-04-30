"""Postflop hand-strength bucketing.

Bucket each (hole, board) state into one of 5 strength categories:
    0: air        — high card, no draw
    1: weak       — bottom pair / weak draw
    2: medium     — middle pair / OESD
    3: strong     — top pair, overpair
    4: nuts-ish   — set, two pair, made flush/straight or better

We use postflop_strength() (already defined in nlhe_player.py) as the
underlying score, with bucket boundaries chosen to roughly split equity space.

For the postflop HU CFR game we also need bucket-vs-bucket equity over
random runouts (turn + river). That's a 5x5 matrix computed by Monte Carlo.

The board state is NOT explicitly part of the bucket — we abstract it
away. This is the standard "imperfect-recall" abstraction used by Cepheus
and many other poker solvers: the bot treats a bucket as a sufficient
statistic for the strategic decision.
"""
from __future__ import annotations

import random
from itertools import combinations

from pokerbot.cache import CACHE_DIR, save_strategy, load_strategy
from pokerbot.core.cards import Card, VALID_RANKS, VALID_SUITS
from pokerbot.core.evaluator import HandEvaluator


NUM_POSTFLOP_BUCKETS = 5

# Boundaries on postflop_strength. Tuned to split roughly evenly:
#   strength < 0.20  -> 0 (air)
#   0.20-0.40        -> 1 (weak)
#   0.40-0.60        -> 2 (medium)
#   0.60-0.80        -> 3 (strong)
#   >= 0.80          -> 4 (nuts)
_BUCKET_BOUNDARIES = (0.20, 0.40, 0.60, 0.80)
_BUCKET_NAMES = ("air", "weak", "medium", "strong", "nuts")


def bucket_for_strength(s: float) -> int:
    """Map a continuous strength score in [0,1] to a bucket index 0..4."""
    for i, b in enumerate(_BUCKET_BOUNDARIES):
        if s < b:
            return i
    return NUM_POSTFLOP_BUCKETS - 1


def bucket_for_hole_and_board(hole: tuple, board: tuple) -> int:
    """Compute bucket given hole cards and current board (3, 4, or 5 cards)."""
    # Lazy import: avoids a circular dependency between
    # pokerbot.abstraction and pokerbot.runtime.
    from pokerbot.runtime.nlhe_player import postflop_strength
    s = postflop_strength(hole, board)
    return bucket_for_strength(s)


def bucket_name(b: int) -> str:
    return _BUCKET_NAMES[b]


def _random_hole_in_bucket(bucket: int, board: list, rng: random.Random,
                           max_attempts: int = 200) -> tuple | None:
    """Sample a hole-card pair that lands in the given bucket on this board.

    Returns None if we can't find one in max_attempts (rare for buckets 0-3,
    common for bucket 4 if the board is dry).
    """
    used = set(board)
    deck = [Card(r, s) for r in VALID_RANKS for s in VALID_SUITS if Card(r, s) not in used]
    for _ in range(max_attempts):
        sample = rng.sample(deck, 2)
        hole = (sample[0], sample[1])
        b = bucket_for_hole_and_board(hole, tuple(board))
        if b == bucket:
            return hole
    return None


def compute_postflop_bucket_equities(
    n_trials: int = 200, seed: int = 0, verbose: bool = True
) -> list:
    """Return a 5x5 matrix where M[i][j] = P(bucket i wins vs bucket j) given a
    random flop and a runout to the river.

    Sampling protocol (per (i, j) pair):
        - Sample a random flop.
        - Find hole cards that put player A in bucket i and player B in bucket j
          on that flop. Skip the flop if either bucket is unreachable.
        - Deal random turn + river.
        - Compare 7-card hands; tally winner.
    """
    n = NUM_POSTFLOP_BUCKETS
    matrix = [[0.5] * n for _ in range(n)]
    if verbose:
        print(f"[abstraction] computing {n}x{n} postflop bucket equity matrix "
              f"({n_trials} MC trials per pair)...")

    rng = random.Random(seed)

    for i in range(n):
        for j in range(n):
            if i == j:
                # Symmetric: by symmetry P(win) ≈ 0.5 (ignoring tie-handling).
                matrix[i][j] = 0.5
                continue

            wins_i = 0.0
            valid_trials = 0
            attempts = 0
            while valid_trials < n_trials and attempts < n_trials * 10:
                attempts += 1
                # Sample a random flop.
                deck = [Card(r, s) for r in VALID_RANKS for s in VALID_SUITS]
                rng.shuffle(deck)
                flop = deck[:3]
                deck = deck[3:]

                # Try to find hole cards in the right buckets.
                # A samples first; B's hole avoids A's cards.
                hole_a = None
                for _ in range(50):
                    sample = rng.sample(deck, 2)
                    if bucket_for_hole_and_board(tuple(sample), tuple(flop)) == i:
                        hole_a = tuple(sample)
                        break
                if hole_a is None:
                    continue
                deck_after_a = [c for c in deck if c not in hole_a]

                hole_b = None
                for _ in range(50):
                    sample = rng.sample(deck_after_a, 2)
                    if bucket_for_hole_and_board(tuple(sample), tuple(flop)) == j:
                        hole_b = tuple(sample)
                        break
                if hole_b is None:
                    continue

                # Deal turn + river.
                deck_after_b = [c for c in deck_after_a if c not in hole_b]
                rng.shuffle(deck_after_b)
                turn = deck_after_b[0]
                river = deck_after_b[1]
                final_board = list(flop) + [turn, river]

                score_a = HandEvaluator.score_seven(list(hole_a) + final_board)
                score_b = HandEvaluator.score_seven(list(hole_b) + final_board)
                if score_a > score_b:
                    wins_i += 1.0
                elif score_a == score_b:
                    wins_i += 0.5
                valid_trials += 1

            if valid_trials > 0:
                matrix[i][j] = wins_i / valid_trials

    return matrix


def cached_postflop_bucket_equities(
    n_trials: int = 200, force: bool = False, verbose: bool = True
) -> list:
    path = CACHE_DIR / f"postflop_bucket_equities_t{n_trials}.pkl"
    if path.exists() and not force:
        if verbose:
            print(f"[cache] loading postflop bucket equities from {path}")
        return load_strategy(path)
    matrix = compute_postflop_bucket_equities(n_trials=n_trials, verbose=verbose)
    save_strategy(matrix, path)
    return matrix
