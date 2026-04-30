"""Disk cache for trained models — save/load CFR strategies and classifiers.

Why we need this: every script run was retraining CFR (~30s for Leduc) and
the classifiers from scratch. With caching, second-and-onward runs of any
script are nearly instant.

Layout: by default everything lives in `.cache/` at the repo root. Each
artifact is a pickle file with a small header dict so we can detect
incompatible artifacts (e.g., wrong file format).

Functions:
    save_strategy(strategy, path)
    load_strategy(path) -> Strategy
    save_classifier(clf, path)
    load_classifier(path) -> Classifier

    cached_cfr_leduc(iterations: int) -> Strategy
        Returns a CFR-trained strategy, training only if not cached.
    cached_classifiers() -> (BluffClassifier, TiltClassifier)
        Same idea — train once, cache, return.

Cache invalidation is by *filename*. Bumping `iterations` changes the
filename, so different training-budget runs don't collide.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

# Root for cached artifacts. Created on first save.
CACHE_DIR = Path(".cache")


_CURRENT_FORMAT = 1


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_strategy(strategy, path) -> None:
    p = Path(path)
    _ensure_dir(p)
    payload = {"format": _CURRENT_FORMAT, "kind": "strategy", "data": strategy}
    with p.open("wb") as f:
        pickle.dump(payload, f)


def load_strategy(path):
    p = Path(path)
    with p.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("kind") != "strategy":
        raise ValueError(f"{path} is not a strategy file (got kind={payload.get('kind')})")
    return payload["data"]


def save_classifier(clf, path) -> None:
    p = Path(path)
    _ensure_dir(p)
    payload = {"format": _CURRENT_FORMAT, "kind": "classifier", "data": clf}
    with p.open("wb") as f:
        pickle.dump(payload, f)


def load_classifier(path):
    p = Path(path)
    with p.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("kind") != "classifier":
        raise ValueError(f"{path} is not a classifier file (got kind={payload.get('kind')})")
    return payload["data"]


# ---- High-level helpers used by scripts ----

def cached_cfr_leduc(iterations: int = 300, force_retrain: bool = False, verbose: bool = True):
    """Return a CFR-trained Leduc strategy, training and caching if needed."""
    from pokerbot.games.leduc import LeducPoker
    from pokerbot.solvers.cfr import CFRSolver

    path = CACHE_DIR / f"cfr_leduc_iters{iterations}.pkl"
    if path.exists() and not force_retrain:
        if verbose:
            print(f"[cache] loading CFR strategy from {path}")
        return load_strategy(path)

    if verbose:
        print(f"[cache] no cached strategy; training CFR for {iterations} iters")
    t0 = time.time()
    game = LeducPoker()
    solver = CFRSolver(game, plus_regret_floor=True, linear_averaging=True)
    solver.train(iterations)
    strategy = solver.average_strategy()
    save_strategy(strategy, path)
    if verbose:
        print(f"[cache] trained + saved in {time.time()-t0:.1f}s -> {path}")
    return strategy


def cached_nlhe_bluff_classifier(
    n_hands: int = 3000,
    epochs: int = 400,
    force_retrain: bool = False,
    verbose: bool = True,
):
    """Train and cache a bluff classifier on NLHE simulator traces."""
    from pokerbot.learning import BluffClassifier
    from pokerbot.runtime.nlhe_features import build_nlhe_bluff_dataset

    path = CACHE_DIR / f"nlhe_bluff_clf_h{n_hands}_e{epochs}.pkl"
    if path.exists() and not force_retrain:
        if verbose:
            print(f"[cache] loading NLHE bluff classifier from {path}")
        return load_classifier(path)

    if verbose:
        print(f"[cache] training NLHE bluff classifier on {n_hands} simulated hands")
    t0 = time.time()
    X, y = build_nlhe_bluff_dataset(n_hands=n_hands, seed=0)
    clf = BluffClassifier().fit(X, y, epochs=epochs)
    save_classifier(clf, path)
    if verbose:
        bluff_rate = (y == 1).mean()
        print(f"[cache] {len(X)} examples ({bluff_rate:.1%} bluffs), trained + saved in "
              f"{time.time() - t0:.1f}s")
    return clf


def cached_preflop_hu_strategy(
    iterations: int = 5000, force_retrain: bool = False, verbose: bool = True
):
    """Train+cache the heads-up preflop CFR strategy."""
    from pokerbot.games.preflop_hu import PreflopHUGame
    from pokerbot.solvers.cfr import CFRSolver
    from pokerbot.abstraction import cached_bucket_equity_matrix

    path = CACHE_DIR / f"preflop_hu_strategy_iter{iterations}.pkl"
    if path.exists() and not force_retrain:
        if verbose:
            print(f"[cache] loading preflop HU strategy from {path}")
        return load_strategy(path)

    if verbose:
        print(f"[cache] training preflop HU CFR for {iterations} iterations")
    t0 = time.time()
    eq_matrix = cached_bucket_equity_matrix(verbose=verbose)
    game = PreflopHUGame(eq_matrix)
    solver = CFRSolver(game, plus_regret_floor=True, linear_averaging=True)
    solver.train(iterations)
    strategy = solver.average_strategy()
    save_strategy(strategy, path)
    if verbose:
        print(f"[cache] trained + saved in {time.time()-t0:.1f}s")
    return strategy


def cached_postflop_hu_strategy(
    iterations: int = 5000, force_retrain: bool = False, verbose: bool = True
):
    """Train+cache the heads-up single-street postflop CFR strategy."""
    from pokerbot.games.postflop_hu import PostflopHUGame
    from pokerbot.solvers.cfr import CFRSolver
    from pokerbot.abstraction import cached_postflop_bucket_equities

    path = CACHE_DIR / f"postflop_hu_strategy_iter{iterations}.pkl"
    if path.exists() and not force_retrain:
        if verbose:
            print(f"[cache] loading postflop HU strategy from {path}")
        return load_strategy(path)

    if verbose:
        print(f"[cache] training postflop HU CFR for {iterations} iterations")
    t0 = time.time()
    eq_matrix = cached_postflop_bucket_equities(verbose=verbose)
    game = PostflopHUGame(eq_matrix)
    solver = CFRSolver(game, plus_regret_floor=True, linear_averaging=True)
    solver.train(iterations)
    strategy = solver.average_strategy()
    save_strategy(strategy, path)
    if verbose:
        print(f"[cache] trained + saved in {time.time()-t0:.1f}s")
    return strategy


def cached_learned_archetypes(
    n_sessions: int = 200,
    hands_per_session: int = 80,
    n_components: int = 5,
    force: bool = False,
    verbose: bool = True,
):
    """Train+cache a GMM-based learned-archetype model.

    Runs simulator sessions, extracts (VPIP, PFR, AGG, WTSD) per player,
    fits a Gaussian Mixture with K components via EM, returns a
    LearnedArchetypes object.
    """
    from pokerbot.opponent import LearnedArchetypes, collect_stat_vectors

    path = CACHE_DIR / f"learned_archetypes_s{n_sessions}_h{hands_per_session}_k{n_components}.pkl"
    if path.exists() and not force:
        if verbose:
            print(f"[cache] loading learned archetypes from {path}")
        return load_classifier(path)

    if verbose:
        print(f"[cache] fitting learned archetypes "
              f"(n_sessions={n_sessions}, hands_per_session={hands_per_session})")
    t0 = time.time()
    X, _ = collect_stat_vectors(
        n_sessions=n_sessions, hands_per_session=hands_per_session,
        seed=0, verbose=verbose,
    )
    if verbose:
        print(f"[cache] collected {len(X)} stat vectors in {time.time()-t0:.1f}s")
        print(f"[cache] fitting GMM with K={n_components} components...")
    archetypes = LearnedArchetypes.fit_from_data(X, n_components=n_components)
    save_classifier(archetypes, path)
    if verbose:
        print(f"[cache] done in {time.time()-t0:.1f}s")
    return archetypes


def cached_real_classifiers(
    n_hands_per_pairing: int = 300,
    epochs: int = 400,
    force_retrain: bool = False,
    verbose: bool = True,
):
    """Train+cache the real-data bluff and tilt classifiers."""
    from pokerbot.learning import BluffClassifier, TiltClassifier
    from pokerbot.runtime import build_bluff_dataset, build_tilt_dataset

    bluff_path = CACHE_DIR / f"bluff_clf_h{n_hands_per_pairing}_e{epochs}.pkl"
    tilt_path = CACHE_DIR / f"tilt_clf_h{n_hands_per_pairing}_e{epochs}.pkl"

    if (
        bluff_path.exists()
        and tilt_path.exists()
        and not force_retrain
    ):
        if verbose:
            print(f"[cache] loading classifiers from {bluff_path.parent}")
        return load_classifier(bluff_path), load_classifier(tilt_path)

    if verbose:
        print("[cache] training real-data classifiers")
    t0 = time.time()
    Xb, yb = build_bluff_dataset(n_hands_per_pairing=n_hands_per_pairing, seed=0)
    Xt, yt = build_tilt_dataset(n_hands_per_pairing=n_hands_per_pairing, seed=0)
    bluff_clf = BluffClassifier().fit(Xb, yb, epochs=epochs)
    tilt_clf = TiltClassifier().fit(Xt, yt, epochs=epochs)
    save_classifier(bluff_clf, bluff_path)
    save_classifier(tilt_clf, tilt_path)
    if verbose:
        print(f"[cache] trained + saved in {time.time()-t0:.1f}s")
    return bluff_clf, tilt_clf


def clear_cache(verbose: bool = True) -> int:
    """Delete every file under CACHE_DIR. Returns number deleted."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for p in CACHE_DIR.glob("*.pkl"):
        p.unlink()
        n += 1
    if verbose:
        print(f"[cache] cleared {n} files")
    return n
