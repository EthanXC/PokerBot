"""Preflop hand bucketing for CFR.

There are exactly 169 distinct preflop hand classes:
    - 13 pocket pairs (22 .. AA)
    - 78 suited unpaired (e.g., A2s..AKs, K2s..KQs, ...)
    - 78 offsuit unpaired

We compute Monte Carlo equity vs. one random opponent for each class, then
sort and bucket into NUM_BUCKETS=10 equal-sized groups. Buckets are indexed
0 (weakest) .. NUM_BUCKETS-1 (strongest).

This is what every poker AI (Cepheus, DeepStack, Pluribus) does for
preflop card abstraction. The resulting bucket map is then used in
`pokerbot.games.preflop_hu` to make the game tree small enough to solve.

All heavy compute (MC equity per class) is cached to disk.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Hashable

from pokerbot.cache import CACHE_DIR, save_strategy, load_strategy
from pokerbot.core.cards import Card, RANK_TO_VALUE, VALID_RANKS, VALID_SUITS
from pokerbot.core.evaluator import HandEvaluator


NUM_BUCKETS = 10


# --- Class identification ---

def _rank_letter(r: str) -> str:
    return "T" if r == "10" else r


def hand_class_str(card1: Card, card2: Card) -> str:
    """Return canonical class string: 'AA', 'AKs', 'AKo', 'TT', 'T9s'."""
    r1 = _rank_letter(card1.rank)
    r2 = _rank_letter(card2.rank)
    suited = card1.suit == card2.suit
    if r1 == r2:
        return f"{r1}{r2}"
    if RANK_TO_VALUE[card1.rank] >= RANK_TO_VALUE[card2.rank]:
        high, low = r1, r2
    else:
        high, low = r2, r1
    return f"{high}{low}{'s' if suited else 'o'}"


def all_169_classes() -> list:
    """Return list of (class_str, representative_card1, representative_card2)
    for each of the 169 distinct preflop hand classes."""
    out = []
    seen = set()
    ranks_desc = sorted(VALID_RANKS, key=lambda r: -RANK_TO_VALUE[r])
    for i, r1 in enumerate(ranks_desc):
        for j, r2 in enumerate(ranks_desc):
            if j < i:
                continue
            if r1 == r2:
                c1, c2 = Card(r1, "S"), Card(r2, "H")
                cls = hand_class_str(c1, c2)
                if cls in seen:
                    continue
                seen.add(cls)
                out.append((cls, c1, c2))
            else:
                c1s, c2s = Card(r1, "S"), Card(r2, "S")
                cls_s = hand_class_str(c1s, c2s)
                if cls_s not in seen:
                    seen.add(cls_s)
                    out.append((cls_s, c1s, c2s))
                c1o, c2o = Card(r1, "S"), Card(r2, "H")
                cls_o = hand_class_str(c1o, c2o)
                if cls_o not in seen:
                    seen.add(cls_o)
                    out.append((cls_o, c1o, c2o))
    return out


# --- Equity computation ---

def equity_class_vs_random(
    card1: Card, card2: Card, n_trials: int = 600, rng: random.Random = None
) -> float:
    """Monte Carlo equity vs 1 random opponent over a 5-card runout."""
    if rng is None:
        rng = random.Random(hash((str(card1), str(card2))) & 0xFFFFFFFF)
    used = {card1, card2}
    deck = [Card(r, s) for r in VALID_RANKS for s in VALID_SUITS if Card(r, s) not in used]
    wins = 0.0
    for _ in range(n_trials):
        sample = rng.sample(deck, 7)
        opp_cards = sample[:2]
        board = list(sample[2:7])
        my_score = HandEvaluator.score_seven([card1, card2] + board)
        opp_score = HandEvaluator.score_seven(opp_cards + board)
        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5
    return wins / n_trials


def compute_class_equities(n_trials: int = 600, verbose: bool = True) -> dict:
    """Compute equity vs random for all 169 classes. Slow (~1-2 minutes)."""
    classes = all_169_classes()
    out = {}
    if verbose:
        print(f"[abstraction] computing equity for {len(classes)} hand classes "
              f"({n_trials} MC trials each)...")
    for i, (cls, c1, c2) in enumerate(classes):
        out[cls] = equity_class_vs_random(c1, c2, n_trials=n_trials)
        if verbose and (i + 1) % 30 == 0:
            print(f"  ... {i+1}/{len(classes)}")
    return out


def make_buckets(class_equities: dict, n_buckets: int = NUM_BUCKETS) -> dict:
    """Sort hand classes by equity, group into n_buckets equal-sized groups.

    Returns {hand_class_str: bucket_index in 0..n_buckets-1}.
    """
    sorted_classes = sorted(class_equities.items(), key=lambda kv: kv[1])
    n = len(sorted_classes)
    bucket_size = (n + n_buckets - 1) // n_buckets
    bucket_map = {}
    for i, (cls, _) in enumerate(sorted_classes):
        b = min(i // bucket_size, n_buckets - 1)
        bucket_map[cls] = b
    return bucket_map


def bucket_for_hand(card1: Card, card2: Card, bucket_map: dict) -> int:
    """Look up which bucket this two-card hand belongs to."""
    return bucket_map[hand_class_str(card1, card2)]


# --- Bucket equity matrix (for showdown in the abstracted game) ---

def compute_bucket_equity_matrix(
    bucket_map: dict, n_trials: int = 200, verbose: bool = True
) -> list:
    """Return n_buckets x n_buckets matrix of P(bucket i wins vs bucket j)."""
    n = NUM_BUCKETS
    # Group hand classes by bucket and pick representative cards from each.
    classes = all_169_classes()
    by_bucket: dict = {}
    for cls, c1, c2 in classes:
        by_bucket.setdefault(bucket_map[cls], []).append((c1, c2))

    matrix = [[0.5] * n for _ in range(n)]
    if verbose:
        print(f"[abstraction] computing {n}x{n} bucket equity matrix...")

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Symmetric matchup: equity = 0.5 + tie correction. We just use 0.5.
                matrix[i][j] = 0.5
                continue
            wins_i = 0.0
            valid = 0
            rng = random.Random(i * 100 + j)
            reps_i = by_bucket.get(i, [])
            reps_j = by_bucket.get(j, [])
            if not reps_i or not reps_j:
                continue
            for _ in range(n_trials):
                cards_i = reps_i[rng.randint(0, len(reps_i) - 1)]
                cards_j = reps_j[rng.randint(0, len(reps_j) - 1)]
                used = set(cards_i) | set(cards_j)
                if len(used) != 4:
                    continue
                deck = [Card(r, s) for r in VALID_RANKS for s in VALID_SUITS
                        if Card(r, s) not in used]
                board = rng.sample(deck, 5)
                si = HandEvaluator.score_seven(list(cards_i) + board)
                sj = HandEvaluator.score_seven(list(cards_j) + board)
                if si > sj:
                    wins_i += 1.0
                elif si == sj:
                    wins_i += 0.5
                valid += 1
            if valid > 0:
                matrix[i][j] = wins_i / valid
                matrix[j][i] = 1.0 - matrix[i][j]
    return matrix


# --- Caching helpers ---

def cached_class_equities(
    n_trials: int = 600, force: bool = False, verbose: bool = True
) -> dict:
    path = CACHE_DIR / f"preflop_class_equities_t{n_trials}.pkl"
    if path.exists() and not force:
        if verbose:
            print(f"[cache] loading class equities from {path}")
        return load_strategy(path)
    equities = compute_class_equities(n_trials=n_trials, verbose=verbose)
    save_strategy(equities, path)
    return equities


def cached_bucket_map(
    n_trials: int = 600, force: bool = False, verbose: bool = True
) -> dict:
    """Return the hand_class_str -> bucket_index mapping."""
    equities = cached_class_equities(n_trials=n_trials, force=force, verbose=verbose)
    return make_buckets(equities)


def cached_bucket_equity_matrix(
    eq_n_trials: int = 600,
    matrix_n_trials: int = 200,
    force: bool = False,
    verbose: bool = True,
) -> list:
    path = CACHE_DIR / f"bucket_equity_matrix_t{matrix_n_trials}.pkl"
    if path.exists() and not force:
        if verbose:
            print(f"[cache] loading bucket equity matrix from {path}")
        return load_strategy(path)
    bm = cached_bucket_map(n_trials=eq_n_trials, force=False, verbose=verbose)
    matrix = compute_bucket_equity_matrix(bm, n_trials=matrix_n_trials, verbose=verbose)
    save_strategy(matrix, path)
    return matrix
