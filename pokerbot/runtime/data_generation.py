"""High-level helpers: 'give me a labeled bluff dataset for training.'

Combines simulator + feature extraction into one call. Mixes opponent
profiles so the dataset isn't biased toward any one playing style.
"""
from __future__ import annotations

import itertools
import random

import numpy as np

from pokerbot.runtime.heuristic_player import PROFILES, make_player
from pokerbot.runtime.session import run_session
from pokerbot.runtime.features import extract_bluff_examples, extract_tilt_examples


def _all_pairs():
    """All ordered pairs of profile names."""
    names = list(PROFILES.keys())
    return list(itertools.product(names, names))


def build_bluff_dataset(
    n_hands_per_pairing: int = 2000,
    seed: int = 0,
) -> tuple:
    """Run sessions across every pair of profiles and collect labeled bluff examples.

    Returns (X, y) ready to feed to BluffClassifier.fit().
    """
    rng = random.Random(seed)
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for p0_name, p1_name in _all_pairs():
        p0 = make_player(p0_name, rng=random.Random(rng.randint(0, 2 ** 31 - 1)))
        p1 = make_player(p1_name, rng=random.Random(rng.randint(0, 2 ** 31 - 1)))
        traces = run_session(
            p0, p1, n_hands_per_pairing,
            rng=random.Random(rng.randint(0, 2 ** 31 - 1)),
        )
        X, y = extract_bluff_examples(traces)
        if len(X):
            all_X.append(X)
            all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


def build_tilt_dataset(
    n_hands_per_pairing: int = 2000,
    seed: int = 0,
) -> tuple:
    """Run sessions and collect labeled tilt examples."""
    rng = random.Random(seed)
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    # Bias toward profiles that tilt (to get enough positives).
    pairs = _all_pairs()
    rng.shuffle(pairs)

    for p0_name, p1_name in pairs:
        p0 = make_player(p0_name, rng=random.Random(rng.randint(0, 2 ** 31 - 1)))
        p1 = make_player(p1_name, rng=random.Random(rng.randint(0, 2 ** 31 - 1)))
        traces = run_session(
            p0, p1, n_hands_per_pairing,
            rng=random.Random(rng.randint(0, 2 ** 31 - 1)),
        )
        X, y = extract_tilt_examples(traces)
        if len(X):
            all_X.append(X)
            all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y
