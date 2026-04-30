"""HumanAwarePolicy — wraps HybridPolicy with bluff + tilt detection.

This is where the AI judgement of human behavior actually changes the bot's
play. Pipeline:

   1. Observe an action by the opponent at info set I.
   2. StrategyMLE updates its posterior over opponent's mixed strategy.
   3. When facing a bet, BluffClassifier scores P(bluff | features).
   4. TiltClassifier scores P(tilt | recent-behavior features).
   5. We construct a "perceived opponent strategy" that:
        - At bet-facing info sets, deviates the opponent's hand range
          toward weaker hands proportionally to P(bluff).
        - When tilt is high, treat the opponent as more reckless (higher
          VPIP/agg) — push hybrid lambda up to extract more EV.
   6. Feed the perceived strategy into HybridPolicy → BR is computed
      vs. that perception → mix with GTO at the appropriate lambda.

The classifiers are decoupled from the game tree — they operate on
features extracted from observations. This means we can train them on
hand histories or synthetic data and plug the trained model in.
"""
from __future__ import annotations

from typing import Hashable

from pokerbot.games.base import ExtensiveFormGame
from pokerbot.policy.hybrid import HybridPolicy
from pokerbot.learning.bluff_classifier import BluffClassifier, FEATURES as BLUFF_FEATURES
from pokerbot.learning.tilt_classifier import TiltClassifier, TILT_FEATURES
from pokerbot.learning.mle_strategy import StrategyMLE


Strategy = dict[Hashable, dict[Hashable, float]]


def adjust_strategy_for_bluff_signal(
    base_strategy: Strategy,
    bet_info_sets: list,
    bluff_prob: float,
) -> Strategy:
    """Deviate the opponent's perceived strategy toward bluffing more at the
    given info sets, proportional to bluff_prob.

    Concretely: at each info set in `bet_info_sets`, mix in extra weight on
    the 'b' (bet/raise) action — implying the opponent is betting wider /
    bluffier. Other info sets pass through unchanged.
    """
    if bluff_prob <= 0:
        return base_strategy
    out = dict(base_strategy)
    for I in bet_info_sets:
        if I not in out:
            continue
        dist = dict(out[I])
        if "b" not in dist:
            continue
        # Push 'b' upward by bluff_prob * (1 - current_b_prob).
        new_b = dist["b"] + bluff_prob * (1 - dist["b"]) * 0.5
        # Renormalize — we proportionally reduce the other actions.
        scale = (1 - new_b) / (1 - dist["b"]) if dist["b"] < 1 else 1.0
        out[I] = {a: (new_b if a == "b" else p * scale) for a, p in dist.items()}
    return out


class HumanAwarePolicy:
    """The full bot: GTO + opponent learning + bluff & tilt classifiers."""

    def __init__(
        self,
        game: ExtensiveFormGame,
        gto_strategy: Strategy,
        player: int,
        bluff_clf: BluffClassifier | None = None,
        tilt_clf: TiltClassifier | None = None,
        base_lambda: float = 0.5,
    ):
        self.game = game
        self.gto = gto_strategy
        self.player = player
        self.base_lambda = base_lambda
        self.bluff_clf = bluff_clf
        self.tilt_clf = tilt_clf
        self.mle = StrategyMLE(prior_alpha=1.0)
        self._hybrid = HybridPolicy(game, gto_strategy, player, base_lambda)

    # --- observation pipeline ---

    def observe_action(self, info_set: Hashable, action: Hashable, legal_actions) -> None:
        """Tell the model about an opponent's action at info_set."""
        self.mle.observe_legal_actions(info_set, legal_actions)
        self.mle.observe(info_set, action)

    # --- decision pipeline ---

    def make_decision(
        self,
        bet_info_sets: list | None = None,
        bluff_features=None,
        tilt_features=None,
    ) -> Strategy:
        """Return the current strategy to play, given (optional) bluff & tilt
        feature inputs.

        - bluff_features: a numpy array (or None) of shape (n_features,).
          If provided AND we have a trained bluff_clf, use the predicted
          P(bluff) to adjust the opponent strategy at `bet_info_sets`.
        - tilt_features: same idea, for tilt detection. High tilt boosts
          our exploit weight.
        """
        modeled = self.mle.strategy(use_map=True)

        # 1. Bluff adjustment: when classifier says high bluff prob, push the
        #    opponent's perceived bet-range wider.
        if (
            self.bluff_clf is not None
            and bluff_features is not None
            and bet_info_sets
        ):
            p_bluff = float(self.bluff_clf.predict_proba(bluff_features)[0])
            modeled = adjust_strategy_for_bluff_signal(modeled, bet_info_sets, p_bluff)
            self._last_bluff_prob = p_bluff
        else:
            self._last_bluff_prob = 0.0

        # 2. Tilt adjustment: when classifier says high tilt prob, raise the
        #    base mixing weight so we exploit harder.
        confidence = self.mle.overall_confidence()
        tilt = 0.0
        if self.tilt_clf is not None and tilt_features is not None:
            tilt = float(self.tilt_clf.predict_proba(tilt_features)[0])
        self._last_tilt_prob = tilt

        # On tilt: bump effective lambda (more exploitation). Off tilt or
        # uncertain: keep the safety guard.
        effective_lambda = self.base_lambda * (1.0 + 0.5 * tilt)
        effective_lambda = min(1.0, effective_lambda)

        # Update hybrid using modeled opponent + effective lambda.
        self._hybrid.base_lambda = effective_lambda
        self._hybrid.update_opponent_model(
            modeled,
            confidence=confidence,
            # we treat tilt as a non-stationarity signal: HIGH tilt usually
            # means the opp's stats are about to swing. We've already used
            # tilt to PUMP the exploit lambda above; now we DON'T let
            # update_opponent_model penalize it.
            tilt=0.0,
        )

        return self._hybrid.strategy()

    @property
    def last_bluff_prob(self) -> float:
        return self._last_bluff_prob

    @property
    def last_tilt_prob(self) -> float:
        return self._last_tilt_prob
