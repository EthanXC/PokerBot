"""AdaptiveBot — the AI human-element layer in actual match-play.

Strategy
--------
Default: play the trained CFR/CFR+ strategy (Nash-ish; safe).

Online deviation: at each decision, build feature vectors from observed
opponent behavior, run the bluff and tilt classifiers, then perturb the
CFR action distribution:

  - If we're FACING a bet/raise and P(bluff) is high  → bump up CALL prob
    (and bump down FOLD); we're bluff-catching wider against perceived bluffs.
  - If P(tilt) is high                                → bump up CALL prob
    on marginal hands; tilted opponents bleed chips, we want to see showdowns.
  - If we are deciding to bet ourselves and we have a STRONG hand AND
    P(opp tilt) is high → upgrade BET → RAISE; extract more from a leaky opp.

The 'amount we deviate' is governed by `deviation_strength` (kept low so
we never go full-exploit and become exploitable ourselves). Empirically,
0.3 - 0.5 works well.

This bot stays decoupled from the CFR solver: it just consumes a strategy
dict.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pokerbot.games.leduc import LeducPoker, card_rank
from pokerbot.learning import BluffClassifier, TiltClassifier
from pokerbot.runtime.feature_builder import (
    BluffFeatureContext,
    build_bluff_feature_vector,
)


@dataclass
class _OppMemory:
    """What the bot remembers about the opponent."""
    recent_results_for_opp: deque = field(default_factory=lambda: deque(maxlen=10))
    n_actions_taken: int = 0
    n_aggressive: int = 0
    n_voluntary: int = 0
    n_hands: int = 0
    hands_since_loss: int = 100
    recent_voluntary: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_aggression: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_loss_streak: int = 0


def _street_index(history: str) -> int:
    return 1 + history.count("/")


def _round_history(history: str) -> str:
    return history.split("/")[-1] if "/" in history else history


def _board_strength_from_card(board_card) -> float:
    if board_card is None:
        return 0.0
    return [0.3, 0.5, 0.7][card_rank(board_card)]


class AdaptiveBotPlayer:
    """A bot that plays CFR by default and deviates when classifiers fire."""

    def __init__(
        self,
        cfr_strategy: dict,
        bluff_clf: BluffClassifier,
        tilt_clf: TiltClassifier,
        deviation_strength: float = 0.25,
        bluff_threshold: float = 0.65,
        tilt_threshold: float = 0.55,
        min_observations: int = 20,
        rng: random.Random | None = None,
    ):
        self.strategy = cfr_strategy
        self.bluff_clf = bluff_clf
        self.tilt_clf = tilt_clf
        self.deviation_strength = deviation_strength
        self.bluff_threshold = bluff_threshold
        self.tilt_threshold = tilt_threshold
        self.min_observations = min_observations
        self.rng = rng or random.Random()
        self._opp = _OppMemory()
        # Per-hand pending counts (flushed in observe_result).
        self._pending_actions = 0
        self._pending_aggr = 0
        # Showdown bookkeeping (used by temporal features 9-12 in v2).
        self._opp_bluff_count = 0
        self._opp_value_count = 0
        self._opp_showdown_count = 0
        # Online-learning buffer: features for opponent bets we've observed
        # this hand. Labeled at showdown when we see opp's card.
        self._pending_bet_features: list = []
        # Whether to nudge the classifier on showdowns:
        self.online_lr: float = 0.01
        self.n_online_updates: int = 0
        # last seen bet so we can extract bluff features when facing it
        self._last_opp_bet = None
        # diagnostic counters
        self.n_decisions = 0
        self.n_deviated = 0

    # --- observation hooks ---

    def observe_result(self, net_chips: float) -> None:
        """Called per-hand by the match runner."""
        self._opp.recent_results_for_opp.append(-net_chips)  # opp's net
        self._opp.n_hands += 1
        if -net_chips < 0:
            self._opp.hands_since_loss = 0
            self._opp.recent_loss_streak += 1
        else:
            self._opp.hands_since_loss += 1
            self._opp.recent_loss_streak = 0
        # Commit any per-hand pending action counts to lifetime totals.
        if self._pending_actions:
            self._opp.n_actions_taken += self._pending_actions
            self._opp.n_aggressive += self._pending_aggr
            self._pending_actions = 0
            self._pending_aggr = 0
        # Clear per-hand pending bet features; they should have been consumed
        # by observe_showdown if showdown occurred, else they're worthless.
        self._pending_bet_features.clear()

    def observe_showdown(self, opp_card: int, board_card) -> None:
        """Called by the match runner when a hand reaches showdown.

        With knowledge of opp's card and the board, we can now LABEL each of
        the opp's bets we recorded this hand and update the classifier
        online. This is real online learning: the bot literally improves
        as it plays.
        """
        from pokerbot.runtime.features import _is_value_or_bluff_for_bettor
        self._opp_showdown_count += 1
        for features_vec, street in self._pending_bet_features:
            board_at_bet = board_card if street == 2 else None
            label = _is_value_or_bluff_for_bettor(opp_card, board_at_bet)
            if label == 1:
                self._opp_bluff_count += 1
            else:
                self._opp_value_count += 1
            # Online classifier update.
            try:
                self.bluff_clf.partial_fit(
                    features_vec[None, :],
                    np.array([label]),
                    lr=self.online_lr,
                )
                self.n_online_updates += 1
            except RuntimeError:
                # Classifier not yet fit; skip.
                pass
        self._pending_bet_features.clear()

    def _observe_opp_action(self, action: str, was_voluntary: bool, was_aggressive: bool) -> None:
        self._opp.n_actions_taken += 1
        if was_aggressive:
            self._opp.n_aggressive += 1

    # --- decision ---

    def decide(self, game: LeducPoker, state, legal_actions: list, my_player: int) -> str:
        self.n_decisions += 1
        cards, history = state
        info_set = game.info_set_key(state, my_player)
        round_h = _round_history(history)
        street = _street_index(history)
        board_card = cards[2] if len(cards) >= 3 else None

        # Track ALL of the opponent's actions in this hand for empirical
        # aggression rate. We re-count from history each time rather than
        # incrementing on the fly (cleaner; avoids double-count bugs).
        opp = 1 - my_player
        full_h = history.replace("/", "")
        opp_actions_this_hand = 0
        opp_aggr_this_hand = 0
        # Walk the history character-by-character and infer actor.
        actor_in_round = 0
        for ch in history:
            if ch == "/":
                actor_in_round = 0
                continue
            if actor_in_round == opp:
                opp_actions_this_hand += 1
                if ch in ("b", "r"):
                    opp_aggr_this_hand += 1
            actor_in_round = 1 - actor_in_round

        # The pending buffers are what we'll flush at end-of-hand in
        # observe_result(). They reflect THIS hand only; we keep them current
        # so observe_result has the correct numbers when called.
        self._pending_actions = opp_actions_this_hand
        self._pending_aggr = opp_aggr_this_hand
        cum_actions = self._opp.n_actions_taken + opp_actions_this_hand
        cum_aggr = self._opp.n_aggressive + opp_aggr_this_hand

        base_dist = self.strategy.get(info_set)
        if base_dist is None:
            return self.rng.choice(legal_actions)

        # Restrict to legal_actions and renormalize.
        items = {a: base_dist.get(a, 0.0) for a in legal_actions}
        total = sum(items.values())
        if total <= 0:
            return self.rng.choice(legal_actions)
        items = {a: p / total for a, p in items.items()}

        # ---- Compute classifier signals ----
        p_bluff = 0.0
        p_tilt = 0.0
        facing_bet = round_h.endswith("b") or round_h.endswith("r")

        # We require enough observations of the opponent before trusting
        # the classifiers — protects us from miscalibration on small samples.
        have_data = cum_actions >= self.min_observations

        # Empirical opp aggression — the gate that prevents us from
        # bluff-catching against tight/passive players regardless of what the
        # per-bet classifier says. Real pro thinking: "they don't bluff enough
        # for the call to pay off, doesn't matter what the board looks like."
        opp_aggression_rate = (
            (cum_aggr + 2.0) / (cum_actions + 7.0) if have_data else 0.0
        )
        opp_is_aggressive = opp_aggression_rate > 0.35

        if facing_bet and have_data and opp_is_aggressive:
            # Build the feature vector via the shared feature_builder so
            # the offline trace extractor and the online bot stay in sync.
            # Convention (matching offline extractor):
            #   n_bets_this_street: count of b/r BEFORE the action being analyzed
            #   decisions_this_street: count INCLUDING the action being analyzed
            n_bets_this_street = sum(1 for ch in round_h[:-1] if ch in ("b", "r"))
            decisions_this_street = len(round_h)
            opp_actions_in_hand = sum(
                1 for i, ch in enumerate(history.replace("/", ""))
                if ch in ("b", "r") and ((i % 2) != my_player)
            ) - (1 if round_h[-1] in ("b", "r") else 0)
            ctx = BluffFeatureContext(
                bet_action=round_h[-1],
                board_card=board_card,
                n_bets_this_street=n_bets_this_street,
                decisions_this_street=decisions_this_street,
                bettor_position=1 - my_player,
                street=street,
                round_history_before=round_h[:-1],
                bettor_aggression_count=cum_aggr,
                bettor_decision_count=cum_actions,
                bettor_bluff_count=self._opp_bluff_count,
                bettor_value_count=self._opp_value_count,
                bettor_showdown_count=self._opp_showdown_count,
                bettor_hand_count=self._opp.n_hands,
                bettor_actions_this_hand=max(0, opp_actions_in_hand),
            )
            features_vec = build_bluff_feature_vector(ctx)
            features = features_vec[None, :]
            p_bluff = float(self.bluff_clf.predict_proba(features)[0])
            # Remember features+context so we can label this bet later
            # at showdown for online learning.
            self._pending_bet_features.append((features_vec.copy(), street))

        if self._opp.n_hands >= 5:
            recent_loss = -min(0.0, sum(self._opp.recent_results_for_opp))
            normalized_loss = min(1.0, recent_loss / 10.0)
            features = np.array([[
                normalized_loss,
                0.0,
                0.0,
                float(self._opp.hands_since_loss),
                0.0,
                float(self._opp.recent_loss_streak),
                0.0,
            ]], dtype=float)
            p_tilt = float(self.tilt_clf.predict_proba(features)[0])

        # ---- Apply deviations to base CFR distribution ----
        # Key change vs naive version: deviation magnitude scales with HOW MUCH
        # the classifier exceeds threshold (not flat above-threshold). This
        # makes us conservative against borderline reads and aggressive
        # against confident ones.
        deviated = False
        new_items = dict(items)

        if facing_bet and have_data:
            # Bluff exploit: shift fold mass to call mass.
            if p_bluff > self.bluff_threshold and "c" in new_items:
                # Excess above threshold, scaled by how much room there is.
                excess = (p_bluff - self.bluff_threshold) / (1 - self.bluff_threshold)
                shift = self.deviation_strength * excess
                fold_p = new_items.get("f", 0.0)
                amt = min(shift, fold_p)
                if amt > 0.001:
                    new_items["f"] = fold_p - amt
                    new_items["c"] = new_items.get("c", 0.0) + amt
                    deviated = True

            # Tilt exploit: small additional call shift (tilted players bet wider).
            if p_tilt > self.tilt_threshold and "c" in new_items:
                excess = (p_tilt - self.tilt_threshold) / (1 - self.tilt_threshold)
                shift = self.deviation_strength * 0.5 * excess
                fold_p = new_items.get("f", 0.0)
                amt = min(shift, fold_p)
                if amt > 0.001:
                    new_items["f"] = fold_p - amt
                    new_items["c"] = new_items.get("c", 0.0) + amt
                    deviated = True

        else:
            # Not facing a bet: if tilt confident, value-extract via raise.
            if p_tilt > self.tilt_threshold and "r" in new_items and "b" in new_items:
                excess = (p_tilt - self.tilt_threshold) / (1 - self.tilt_threshold)
                shift = self.deviation_strength * 0.5 * excess
                bet_p = new_items.get("b", 0.0)
                amt = min(shift, bet_p)
                if amt > 0.001:
                    new_items["b"] = bet_p - amt
                    new_items["r"] = new_items.get("r", 0.0) + amt
                    deviated = True

        if deviated:
            self.n_deviated += 1

        # Renormalize and sample.
        total = sum(new_items.values())
        if total <= 0:
            return self.rng.choice(legal_actions)
        new_items = {a: p / total for a, p in new_items.items()}
        r = self.rng.random()
        cum = 0.0
        for a, p in new_items.items():
            cum += p
            if r < cum:
                return a
        # Numerical fallback
        return list(new_items.keys())[-1]
