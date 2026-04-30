"""Learned opponent archetypes via EM-fitted Gaussian Mixture Model.

Replaces (or augments) the hand-coded archetypes in `archetypes.py` with
clusters DISCOVERED from data. The pipeline:

  1. Run NLHE simulator sessions across all heuristic profiles.
  2. For each player in each session, compute a 4-dim stat vector
     (VPIP, PFR, AGG, WTSD) over their hands.
  3. Fit a Gaussian Mixture Model with K=5 components via EM.
  4. The 5 discovered clusters correspond (loosely) to canonical archetypes
     — the model rediscovers them without being told.

The bot uses the cluster posterior P(z = k | stats) to modulate
deviation_strength per-opponent: against an "exploitable" cluster
(high VPIP + high AGG), it deviates more aggressively.

Course topics covered: Expectation Maximization, Gaussian Mixture Models,
unsupervised learning.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Hashable

import numpy as np

from pokerbot.cache import CACHE_DIR, save_strategy, load_strategy
from pokerbot.games.nlhe import (
    NLHE, NLHEConfig,
    FOLD, CHECK, CALL, POT_BET, ALL_IN,
)
from pokerbot.learning.gmm import fit_gmm, GaussianMixture


# Stats we extract per player. Kept small (4) to stay learnable from
# moderate amounts of data.
STAT_NAMES = ("VPIP", "PFR", "AGG", "WTSD")
STAT_DIM = len(STAT_NAMES)


def _stat_vector_from_session_log(
    n_players: int, history_per_hand: list, hole_cards_per_hand: list
) -> list:
    """Compute a (VPIP, PFR, AGG, WTSD) vector for each seat from a session log.

    history_per_hand: list of (history tuple-of-tuples-of-(actor,action), folded set)
    """
    n_hands = [0] * n_players
    n_vpip = [0] * n_players
    n_pfr = [0] * n_players
    n_agg = [0] * n_players
    n_decisions = [0] * n_players  # for AGG denominator
    n_saw_flop = [0] * n_players
    n_wtsd = [0] * n_players

    for hand_history, folded, ended_at_round in history_per_hand:
        # Each player who isn't dealt-in for free (i.e., everyone) counts as a hand.
        for s in range(n_players):
            n_hands[s] += 1

        # Track preflop voluntary action and aggressive action per seat for THIS hand.
        seat_vpip_this = [False] * n_players
        seat_pfr_this = [False] * n_players

        for round_idx, round_actions in enumerate(hand_history):
            for actor, action in round_actions:
                n_decisions[actor] += 1
                if action in (POT_BET, ALL_IN):
                    n_agg[actor] += 1
                    if round_idx == 0:
                        seat_pfr_this[actor] = True
                if round_idx == 0 and action in (CALL, POT_BET, ALL_IN):
                    seat_vpip_this[actor] = True

        for s in range(n_players):
            if seat_vpip_this[s]:
                n_vpip[s] += 1
            if seat_pfr_this[s]:
                n_pfr[s] += 1

        # Saw flop: any action in round 1+
        for s in range(n_players):
            if any(actor == s for r in hand_history[1:] for actor, _ in r):
                n_saw_flop[s] += 1

        # WTSD: didn't fold AND hand reached river closure
        for s in range(n_players):
            if s not in folded and ended_at_round >= 3:
                n_wtsd[s] += 1

    vectors = []
    for s in range(n_players):
        if n_hands[s] < 20:
            # Skip seats with too little data.
            continue
        vpip = n_vpip[s] / n_hands[s]
        pfr = n_pfr[s] / n_hands[s]
        agg = n_agg[s] / max(1, n_decisions[s])
        wtsd = n_wtsd[s] / max(1, n_saw_flop[s])
        vectors.append([vpip, pfr, agg, wtsd])
    return vectors


def _play_session_for_stats(
    profile_names: list, n_hands: int, rng: random.Random,
) -> list:
    """Run one NLHE session and return per-seat stat vectors."""
    # Lazy imports to avoid circular dependency between opponent and runtime.
    from pokerbot.runtime import make_nlhe_player
    from pokerbot.runtime.nlhe_match import _shuffle_deck

    n_players = len(profile_names)
    config = NLHEConfig(n_players=n_players)
    game = NLHE(config)
    players = [
        make_nlhe_player(p, rng=random.Random(rng.randint(0, 2 ** 31 - 1)))
        for p in profile_names
    ]

    history_per_hand: list = []
    for hand_idx in range(n_hands):
        deck = _shuffle_deck(rng)
        button = hand_idx % n_players
        state = game.initial_state(deck_order=deck, button=button)
        steps = 0
        while not game.is_terminal(state) and steps < 500:
            if game.is_chance(state):
                outcomes = game.chance_outcomes(state)
                r = rng.random()
                cum = 0.0
                chosen = outcomes[-1][0]
                for a, p in outcomes:
                    cum += p
                    if r < cum:
                        chosen = a
                        break
                state = game.apply(state, chosen)
            else:
                seat = state.actor
                legal = game.legal_actions(state)
                action = players[seat].decide(game, state, legal, seat)
                if action not in legal:
                    action = legal[0]
                state = game.apply(state, action)
            steps += 1
        # Record per-hand log: (history, folded set, round_idx_at_end).
        history_per_hand.append((state.history, state.folded, state.round_idx))
        # Tilt-mechanic feedback to players.
        for s in range(n_players):
            net = game.utility(state, s)
            if hasattr(players[s], "observe_result"):
                players[s].observe_result(net)

    return _stat_vector_from_session_log(n_players, history_per_hand, None)


def collect_stat_vectors(
    n_sessions: int = 250,
    hands_per_session: int = 80,
    seed: int = 0,
    verbose: bool = True,
) -> tuple:
    """Run sessions and return (stat_matrix, profile_labels).

    profile_labels are kept for diagnostic use only — the GMM never sees them.
    """
    # Lazy import (avoid circular dep at module load).
    from pokerbot.runtime import NLHE_PROFILES

    rng = random.Random(seed)
    profile_names = list(NLHE_PROFILES.keys())
    table_sizes = [2, 3, 4, 6]

    all_vectors = []
    all_profile_labels = []

    if verbose:
        print(f"[learned-archetypes] running {n_sessions} sessions "
              f"(~{hands_per_session} hands each)...")
    for s_idx in range(n_sessions):
        n_players = rng.choice(table_sizes)
        chosen = [rng.choice(profile_names) for _ in range(n_players)]
        vectors = _play_session_for_stats(chosen, hands_per_session, rng)
        all_vectors.extend(vectors)
        all_profile_labels.extend(chosen[:len(vectors)])
        if verbose and (s_idx + 1) % max(1, n_sessions // 10) == 0:
            print(f"  ... {s_idx+1}/{n_sessions} sessions, {len(all_vectors)} vectors")
    return np.array(all_vectors), all_profile_labels


# ---- LearnedArchetypes class ----

@dataclass
class LearnedArchetypes:
    gmm: GaussianMixture
    feature_names: tuple = STAT_NAMES

    @classmethod
    def fit_from_data(cls, X: np.ndarray, n_components: int = 5, seed: int = 0):
        gmm = fit_gmm(X, n_components=n_components, seed=seed, max_iters=200)
        return cls(gmm=gmm)

    @property
    def n_components(self) -> int:
        return self.gmm.n_components

    @property
    def cluster_means(self) -> np.ndarray:
        return self.gmm.means

    @property
    def cluster_weights(self) -> np.ndarray:
        return self.gmm.weights

    def responsibility(self, stat_vec: np.ndarray) -> np.ndarray:
        """Return P(z = k | stat_vec) for each cluster. Shape (K,)."""
        if stat_vec.ndim == 1:
            stat_vec = stat_vec[None, :]
        return self.gmm.responsibilities(stat_vec)[0]

    def exploitability_score_per_cluster(self) -> np.ndarray:
        """A scalar in [0, 1] per cluster: how exploitable is this archetype?

        Heuristic: clusters with HIGH VPIP and HIGH AGG are exploitable
        (loose-aggressive players bleed chips). LOW VPIP + LOW AGG (nits)
        are tough to exploit.

        Score = 0.6 * normalized(AGG) + 0.4 * normalized(VPIP)
        normalized via min-max over clusters.
        """
        means = self.cluster_means
        # Indices: 0=VPIP, 1=PFR, 2=AGG, 3=WTSD
        vpip = means[:, 0]
        agg = means[:, 2]

        def _normalize(x):
            lo, hi = x.min(), x.max()
            if hi - lo < 1e-9:
                return np.zeros_like(x)
            return (x - lo) / (hi - lo)

        return 0.6 * _normalize(agg) + 0.4 * _normalize(vpip)

    def opponent_exploitability(self, stat_vec: np.ndarray) -> float:
        """Weighted exploitability for an opponent given their observed stats."""
        r = self.responsibility(stat_vec)
        scores = self.exploitability_score_per_cluster()
        return float(r @ scores)
