"""MultiOpponentAdaptiveBot — N-player extension of AdaptiveBotPlayer.

Same idea as the heads-up version, generalized:
  - A separate stats tracker PER OPPONENT seat.
  - When facing a bet from seat X, the bluff classifier evaluates X's bet
    using X's stats (X-specific aggression rate, X-specific bluff history).
  - Tilt is also per-opponent.
  - The decision policy:
      * If the LATEST bettor's bluff probability is high enough AND that
        opponent has empirically been aggressive, increase our call probability.
      * If the latest bettor is on tilt, weight calling/raising-for-value more.
      * Otherwise, fall back to a simple GTO-ish heuristic
        (postflop_strength based decision via the same logic as the
        heuristic player, but using a "balanced" profile).

Why heuristic baseline (not CFR): a CFR-trained NLHE strategy at our scale
would be very weak — Pluribus needed weeks of compute. The heuristic
"balanced" baseline plus classifier-driven deviations is sufficient to
demonstrate the architecture and beat exploitable opponents.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pokerbot.games.nlhe import (
    NLHE, NLHEState,
    FOLD, CHECK, CALL, POT_BET, ALL_IN,
)
from pokerbot.learning import BluffClassifier, TiltClassifier
from pokerbot.runtime.nlhe_player import (
    NLHEHeuristicPlayer, PROFILES,
    preflop_strength, postflop_strength,
)
from pokerbot.abstraction import bucket_for_hand, bucket_for_hole_and_board
from pokerbot.opponent.learned_archetypes import LearnedArchetypes


@dataclass
class _PerSeatStats:
    """What the bot remembers about ONE opponent seat."""
    n_actions_taken: int = 0
    n_aggressive: int = 0
    n_hands: int = 0
    n_showdowns: int = 0
    n_bluffs_revealed: int = 0
    n_value_revealed: int = 0
    # For VPIP/PFR (LearnedArchetypes feature vector)
    n_vpip: int = 0       # hands where opp voluntarily put $ in preflop
    n_pfr: int = 0        # hands where opp raised preflop
    n_saw_flop: int = 0   # hands where opp made it to round 1+
    recent_results: deque = field(default_factory=lambda: deque(maxlen=10))
    hands_since_loss: int = 100
    loss_streak: int = 0


class MultiOpponentAdaptiveBot:
    """N-player adaptive bot with per-opponent classifier tracking."""

    def __init__(
        self,
        bluff_clf: BluffClassifier,
        tilt_clf: TiltClassifier,
        baseline_profile: str = "tight_passive",
        deviation_strength: float = 0.30,
        bluff_threshold: float = 0.65,
        tilt_threshold: float = 0.55,
        min_observations: int = 15,
        preflop_hu_strategy: dict | None = None,
        bucket_map: dict | None = None,
        postflop_hu_strategy: dict | None = None,
        use_preflop_chart: bool = True,
        learned_archetypes: LearnedArchetypes | None = None,
        rng: random.Random | None = None,
    ):
        self.bluff_clf = bluff_clf
        self.tilt_clf = tilt_clf
        # Heuristic fallback (used postflop and in multi-way pots).
        self._baseline = NLHEHeuristicPlayer(
            PROFILES[baseline_profile],
            rng=rng or random.Random(),
        )
        # Game-theoretic preflop blueprint (heads-up only, bucket abstraction).
        # When provided, overrides the heuristic for HU preflop spots.
        self.preflop_hu_strategy = preflop_hu_strategy
        self.bucket_map = bucket_map
        # Game-theoretic POSTFLOP blueprint (heads-up only, single-street).
        # When provided, overrides the heuristic for HU flop+ decisions.
        self.postflop_hu_strategy = postflop_hu_strategy
        # Preflop chart: simple threshold-based opening, replaces preflop CFR
        # (since preflop is well-charted in poker theory; the value-add of
        # CFR is bigger postflop where boards/sizes interact).
        self.use_preflop_chart = use_preflop_chart
        # Optional GMM-based archetype model. When set, the bot scales its
        # deviation strength per-opponent according to that opponent's
        # exploitability score.
        self.learned_archetypes = learned_archetypes
        self.deviation_strength = deviation_strength
        self.bluff_threshold = bluff_threshold
        self.tilt_threshold = tilt_threshold
        self.min_observations = min_observations
        self.rng = rng or random.Random()
        # Per-opponent state, populated lazily as we see new seats.
        self._stats: dict = {}
        # Diagnostic counters.
        self.n_decisions = 0
        self.n_deviated = 0
        self.n_online_updates = 0
        self.n_preflop_cfr_lookups = 0
        self.n_postflop_cfr_lookups = 0
        self.n_preflop_chart_lookups = 0

    # --- bookkeeping ---

    def _seat(self, idx: int) -> _PerSeatStats:
        if idx not in self._stats:
            self._stats[idx] = _PerSeatStats()
        return self._stats[idx]

    def _opponent_stat_vector(self, seat: int) -> np.ndarray | None:
        """Return [VPIP, PFR, AGG, WTSD] for this opponent, or None if no data."""
        s = self._stats.get(seat)
        if s is None or s.n_hands < 5:
            return None
        vpip = s.n_vpip / s.n_hands
        pfr = s.n_pfr / s.n_hands
        agg = s.n_aggressive / max(1, s.n_actions_taken)
        wtsd = s.n_showdowns / max(1, s.n_saw_flop)
        return np.array([vpip, pfr, agg, wtsd], dtype=float)

    def _opponent_exploitability_score(self, seat: int) -> float:
        """Use the GMM-learned archetype model to score how exploitable this
        opponent is. Returns 1.0 (default behavior) when:
          - no learned_archetypes is configured
          - we don't have enough observations of this opponent yet

        Otherwise returns a value in [0, 1] proportional to the cluster-
        weighted exploitability score.
        """
        if self.learned_archetypes is None:
            return 1.0
        x = self._opponent_stat_vector(seat)
        if x is None:
            return 1.0
        # Map [0, 1] from the GMM into a meaningful multiplier.
        # We use 0.4 + 0.8 * raw_score so even "tight" opps get SOME deviation
        # (the classifier has its own threshold gate above) while bluffy opps
        # get up to 1.2x.
        raw = self.learned_archetypes.opponent_exploitability(x)
        return 0.4 + 0.8 * raw

    def observe_result(self, net_chips: float) -> None:
        # Simulator-level hook — for OUR seat. The match harness calls this
        # with the bot's own net. We don't use it here directly; opponent
        # stats are derived per-hand below from the action history.
        pass

    def _update_stats_from_history(self, state: NLHEState, my_player: int) -> None:
        """Refresh per-seat aggression counters from the public history."""
        # Reset and re-walk; cheap because per-hand history is short.
        n = state.config.n_players
        for seat in range(n):
            if seat == my_player:
                continue
            stats = self._seat(seat)
            actions_seen = 0
            aggressive_seen = 0
            for round_h in state.history:
                for actor, action in round_h:
                    if actor != seat:
                        continue
                    actions_seen += 1
                    if action in (POT_BET, ALL_IN):
                        aggressive_seen += 1
            # We track per-hand (this hand) counts; lifetime counters get
            # incremented in observe_hand_finalized via the match harness.
            stats._this_hand_actions = actions_seen
            stats._this_hand_aggressive = aggressive_seen

    # --- decision ---

    def _try_preflop_chart_action(self, state: NLHEState, legal_actions: list,
                                   my_player: int) -> str | None:
        """Naive preflop chart: simple thresholds on preflop_strength.

        This replaces a CFR-trained preflop strategy. Preflop is well-known
        from theory and easy to chart; we save our CFR budget for postflop.
        """
        if state.round_idx != 0:
            return None

        my_hole = state.hole_cards[my_player]
        s = preflop_strength(my_hole)

        owed = state.bet_to_match - state.contributed_this_round[my_player]
        facing_bet = owed > 0

        if facing_bet:
            # Facing a raise. Threshold for continuing.
            if s < 0.45:
                if FOLD in legal_actions:
                    return FOLD
                return CALL if CALL in legal_actions else legal_actions[0]
            if s > 0.78:
                # Strong: 3-bet
                if POT_BET in legal_actions:
                    return POT_BET
                if CALL in legal_actions:
                    return CALL
            if CALL in legal_actions:
                return CALL
            return legal_actions[0]
        else:
            # First-to-act. Open-raise with strong, limp/check with medium.
            if s < 0.42:
                # Too weak to play. Check if free, fold otherwise.
                if CHECK in legal_actions:
                    return CHECK
                if FOLD in legal_actions:
                    return FOLD
            if s > 0.55:
                if POT_BET in legal_actions:
                    return POT_BET
            if CALL in legal_actions:
                return CALL
            if CHECK in legal_actions:
                return CHECK
            return legal_actions[0]

    def _try_postflop_cfr_action(self, state: NLHEState, legal_actions: list,
                                  my_player: int) -> str | None:
        """Look up an action from the postflop CFR blueprint.

        ASYMMETRIC TRUST: we only return a CFR action when it is AGGRESSIVE
        (BET or RAISE). The CFR strategy lives in a heavily abstracted game
        and tends to under-bet OOP relative to real NLHE; we don't trust its
        passive lines but we DO trust its aggressive ones (since aggressive
        moves are usually value-driven and survive most abstractions).

        For passive decisions (CHECK/CALL/FOLD), we return None so the bot
        falls back to its heuristic baseline.
        """
        if (
            self.postflop_hu_strategy is None
            or state.round_idx == 0
            or len(state.board) < 3
        ):
            return None
        n = state.config.n_players
        active = [s for s in range(n) if s not in state.folded]
        if len(active) != 2 or my_player not in active:
            return None

        # Compute my bucket on this board.
        try:
            my_bucket = bucket_for_hole_and_board(
                state.hole_cards[my_player], tuple(state.board)
            )
        except Exception:
            return None

        # Translate the current round's history to the abstracted action language.
        from pokerbot.games.postflop_hu import (
            FOLD as PF_FOLD, CHECK as PF_CHECK, CALL as PF_CALL,
            BET as PF_BET, RAISE as PF_RAISE,
        )
        round_h = state.history[state.round_idx]
        # Determine if there's been a bet in THIS round so we know whether
        # POT_BET means "open bet" or "raise".
        had_bet = False
        hist_str = ""
        for seat, action in round_h:
            if action == FOLD:
                hist_str += PF_FOLD
                break
            elif action == CHECK:
                hist_str += PF_CHECK
            elif action == CALL:
                hist_str += PF_CALL
            elif action in (POT_BET, ALL_IN):
                hist_str += PF_RAISE if had_bet else PF_BET
                had_bet = True

        info_set_key = f"pf{my_bucket}|{hist_str}"
        action_dist = self.postflop_hu_strategy.get(info_set_key)
        if action_dist is None:
            return None

        # Aggregate the aggressive mass — total probability of bet+raise.
        aggressive_prob = action_dist.get(PF_BET, 0.0) + action_dist.get(PF_RAISE, 0.0)
        # Only consult CFR if it's confidently aggressive in this spot.
        # Threshold of 0.30 means "CFR thinks this spot is at least somewhat
        # bet-worthy"; below that we let the heuristic decide.
        if aggressive_prob < 0.30:
            return None

        # Sample within the aggressive subset (renormalize).
        r = self.rng.random() * aggressive_prob
        cum = 0.0
        chosen_abstract = None
        for a in (PF_BET, PF_RAISE):
            p = action_dist.get(a, 0.0)
            cum += p
            if r < cum:
                chosen_abstract = a
                break
        if chosen_abstract is None:
            chosen_abstract = PF_BET

        abstract_to_nlhe = {
            PF_BET: POT_BET,
            PF_RAISE: POT_BET,
        }
        nlhe_action = abstract_to_nlhe.get(chosen_abstract)
        if nlhe_action is None or nlhe_action not in legal_actions:
            if ALL_IN in legal_actions:
                nlhe_action = ALL_IN
            else:
                return None

        self.n_postflop_cfr_lookups += 1
        return nlhe_action

    def _try_preflop_cfr_action(self, state: NLHEState, legal_actions: list,
                                 my_player: int) -> str | None:
        """If we're in a heads-up preflop spot AND have the CFR strategy,
        look up an action from the GTO blueprint.

        Returns None if we should fall back to the heuristic baseline.
        """
        if (
            self.preflop_hu_strategy is None
            or self.bucket_map is None
            or state.round_idx != 0
        ):
            return None

        # HU = exactly 2 non-folded players.
        n = state.config.n_players
        active = [s for s in range(n) if s not in state.folded]
        if len(active) != 2 or my_player not in active:
            return None

        # Translate the NLHE preflop history to the abstracted (f/c/r) language.
        # In our abstracted game, only my_player and one opponent matter.
        hist_str = ""
        my_role = None  # 0 if I'm SB, 1 if BB
        # In NLHE the SB seat is button+1 (3+ players) or button (HU).
        # For >2 players, "HU preflop" only happens after others fold to me, but
        # before that point folds are part of the history. We extract just MY
        # actions and the LAST aggressor's actions to roughly map to the game.
        # Simpler: just use the heuristic in non-pure-HU spots.
        if n != 2:
            return None
        my_role = 0 if my_player == state.button else 1

        # Build action history of just SB and BB, mapping
        #   POT_BET -> "r", CALL -> "c", FOLD -> "f", CHECK -> "c", ALL_IN -> "r"
        # This is approximate but sufficient for blueprint lookup.
        for seat, action in state.history[0]:
            if action in (POT_BET, ALL_IN):
                hist_str += "r"
            elif action in (CALL, CHECK):
                hist_str += "c"
            elif action == FOLD:
                hist_str += "f"
                break

        # Look up our bucket.
        my_hole = state.hole_cards[my_player]
        try:
            my_bucket = bucket_for_hand(my_hole[0], my_hole[1], self.bucket_map)
        except KeyError:
            return None
        info_set_key = f"b{my_bucket}|{hist_str}"

        action_dist = self.preflop_hu_strategy.get(info_set_key)
        if action_dist is None:
            return None

        # Sample an action from the distribution. Then translate back to NLHE actions.
        r = self.rng.random()
        cum = 0.0
        chosen_abstract = None
        for a, p in action_dist.items():
            cum += p
            if r < cum:
                chosen_abstract = a
                break
        if chosen_abstract is None:
            chosen_abstract = list(action_dist.keys())[-1]

        # Translate.
        abstract_to_nlhe = {
            "f": FOLD,
            "c": CALL if CALL in legal_actions else CHECK,
            "r": POT_BET,
        }
        nlhe_action = abstract_to_nlhe.get(chosen_abstract)
        if nlhe_action is None or nlhe_action not in legal_actions:
            # Maybe POT_BET isn't legal (cap reached); try ALL_IN as escalation.
            if chosen_abstract == "r" and ALL_IN in legal_actions:
                nlhe_action = ALL_IN
            else:
                return None

        self.n_preflop_cfr_lookups += 1
        return nlhe_action

    def decide(self, game: NLHE, state: NLHEState, legal_actions: list, my_player: int) -> str:
        self.n_decisions += 1
        self._update_stats_from_history(state, my_player)

        # 0. Decision pipeline:
        #    Preflop  → simple chart (naive)
        #    Postflop → CFR blueprint (heads-up, bucket abstraction)
        #    Otherwise → heuristic
        baseline_action = None
        if state.round_idx == 0 and self.use_preflop_chart:
            baseline_action = self._try_preflop_chart_action(state, legal_actions, my_player)
            if baseline_action is not None:
                self.n_preflop_chart_lookups += 1
        elif state.round_idx == 0 and self.preflop_hu_strategy is not None:
            baseline_action = self._try_preflop_cfr_action(state, legal_actions, my_player)
        elif state.round_idx >= 1 and self.postflop_hu_strategy is not None:
            baseline_action = self._try_postflop_cfr_action(state, legal_actions, my_player)

        if baseline_action is None:
            # Fall back to the heuristic baseline.
            baseline_action = self._baseline.decide(game, state, legal_actions, my_player)

        # 2. Identify the most recent aggressor (whose bet we might be bluff-catching).
        last_aggressor = None
        last_action = None
        for round_h in reversed(state.history):
            for actor, action in reversed(round_h):
                if action in (POT_BET, ALL_IN) and actor != my_player:
                    last_aggressor = actor
                    last_action = action
                    break
            if last_aggressor is not None:
                break

        owed = state.bet_to_match - state.contributed_this_round[my_player]
        facing_bet = owed > 0

        # 3. If we're facing a bet AND have data on the aggressor, run the
        #    bluff classifier to decide whether to deviate from baseline.
        deviated = False
        action = baseline_action

        if facing_bet and last_aggressor is not None:
            stats = self._seat(last_aggressor)
            cum_actions = stats.n_actions_taken + stats._this_hand_actions
            cum_aggr = stats.n_aggressive + stats._this_hand_aggressive
            have_data = cum_actions >= self.min_observations

            if have_data:
                # Empirical aggression-rate gate. NLHE-calibrated:
                # most non-bluffy NLHE players bet/raise <8% of decisions,
                # bluffy ones (loose_aggressive, maniac) >12%.
                opp_aggression_rate = (cum_aggr + 2.0) / (cum_actions + 7.0)
                opp_is_aggressive = opp_aggression_rate > 0.12

                if opp_is_aggressive:
                    p_bluff = self._compute_bluff_prob(state, last_aggressor, last_action, my_player)
                    p_tilt = self._compute_tilt_prob(last_aggressor)

                    # If both signals say we should be calling more, do so.
                    if (p_bluff > self.bluff_threshold or p_tilt > self.tilt_threshold) \
                       and baseline_action == FOLD and CALL in legal_actions:
                        # Convert the FOLD into a CALL with probability proportional
                        # to how confident we are.
                        confidence = max(
                            (p_bluff - self.bluff_threshold) if p_bluff > self.bluff_threshold else 0.0,
                            (p_tilt - self.tilt_threshold) if p_tilt > self.tilt_threshold else 0.0,
                        ) / 0.4  # normalize to roughly [0, 1]
                        confidence = min(1.0, confidence)
                        # Per-opponent exploitability scaling (from learned GMM
                        # archetypes). If we don't have the model, defaults to 1.0.
                        exploit_scale = self._opponent_exploitability_score(last_aggressor)
                        flip_prob = self.deviation_strength * confidence * exploit_scale
                        if self.rng.random() < flip_prob:
                            action = CALL
                            deviated = True

        if deviated:
            self.n_deviated += 1

        # Final safety: ensure action is legal.
        if action not in legal_actions:
            action = legal_actions[0]
        return action

    # --- classifier feature construction ---

    def _compute_bluff_prob(self, state: NLHEState, bettor: int, last_action: str, my_player: int) -> float:
        """Build the 12-feature bluff vector and run the classifier."""
        stats = self._seat(bettor)
        cum_actions = stats.n_actions_taken + stats._this_hand_actions
        cum_aggr = stats.n_aggressive + stats._this_hand_aggressive

        # Find bets/raises this street so far.
        round_h = state.history[state.round_idx]
        n_bets_this_street = sum(1 for _, a in round_h if a in (POT_BET, ALL_IN))
        decisions_this_street = len(round_h)

        # Map NLHE board strength into our 0..1 feature space.
        # (Use postflop_strength on a "phantom hand" with two random low cards
        # as a proxy for "how connected/dangerous is the board.")
        if state.board:
            from pokerbot.core.cards import Card
            # Use a low non-card-set proxy for board "danger"
            phantom = (Card("2", "C"), Card("3", "D"))
            # Avoid card collisions if board contains them
            board_cards = list(state.board)
            board_str = postflop_strength(phantom, tuple(board_cards))
        else:
            board_str = 0.0

        # Pot committed (proxy via this-street contributions).
        pot_committed = min(1.0, state.contributed[bettor] / max(1, state.config.starting_stack))

        # Bluff/value rates from showdown observations.
        bluff_rate = (stats.n_bluffs_revealed + 1.0) / (stats.n_bluffs_revealed + stats.n_value_revealed + 2.0)
        showdown_rate = (stats.n_showdowns + 1.0) / (stats.n_hands + 2.0)

        # Aggressor's actions earlier in this hand (sum across rounds for this seat).
        bettor_hand_aggr = sum(
            1 for r in state.history for actor, a in r
            if actor == bettor and a in (POT_BET, ALL_IN)
        ) - (1 if last_action in (POT_BET, ALL_IN) else 0)

        prior_aggression = (cum_aggr + 2.0) / (cum_actions + 7.0)

        features = np.array([[
            1.5 if last_action == ALL_IN else 1.0,    # bet_size_ratio
            board_str,                                 # board_strength
            prior_aggression,                          # prior_aggression
            1.0 if (bettor > my_player) else 0.0,      # position
            pot_committed,
            n_bets_this_street,
            1.0 if last_action == ALL_IN else 0.0,     # overbet
            0.0,                                        # donk_lead (no preflop raiser concept used here)
            bluff_rate,
            showdown_rate,
            float(max(0, bettor_hand_aggr)),
            n_bets_this_street / max(1, decisions_this_street),
        ]], dtype=float)

        return float(self.bluff_clf.predict_proba(features)[0])

    def _compute_tilt_prob(self, bettor: int) -> float:
        stats = self._seat(bettor)
        if stats.n_hands < 5:
            return 0.0
        recent_loss = -min(0.0, sum(stats.recent_results))
        normalized_loss = min(1.0, recent_loss / 30.0)  # NLHE scale: $30 ~= sizable hit
        features = np.array([[
            normalized_loss,
            0.0, 0.0,
            float(stats.hands_since_loss),
            0.0,
            float(stats.loss_streak),
            0.0,
        ]], dtype=float)
        return float(self.tilt_clf.predict_proba(features)[0])

    # --- end-of-hand hook (called by play_table via observe_hand_finalized) ---

    def observe_hand_finalized(self, state: NLHEState, my_player: int, all_nets: list) -> None:
        """Update per-seat lifetime stats given the resolved hand."""
        n = state.config.n_players
        went_to_showdown = "f" not in "".join(
            "".join(a for _, a in r) for r in state.history
        )
        for seat in range(n):
            if seat == my_player:
                continue
            stats = self._seat(seat)
            stats.n_actions_taken += stats._this_hand_actions
            stats.n_aggressive += stats._this_hand_aggressive
            stats.n_hands += 1
            stats.recent_results.append(all_nets[seat])
            if all_nets[seat] < 0:
                stats.hands_since_loss = 0
                stats.loss_streak += 1
            else:
                stats.hands_since_loss += 1
                stats.loss_streak = 0
            if went_to_showdown:
                stats.n_showdowns += 1
            # VPIP and PFR per hand (parse round 0 actions for this seat).
            had_voluntary_action = False
            had_pfr = False
            for actor, action in state.history[0]:
                if actor != seat:
                    continue
                if action in (CALL, POT_BET, ALL_IN):
                    had_voluntary_action = True
                if action in (POT_BET, ALL_IN):
                    had_pfr = True
            if had_voluntary_action:
                stats.n_vpip += 1
            if had_pfr:
                stats.n_pfr += 1
            # Saw flop: any action in round 1+
            if any(actor == seat for r in state.history[1:] for actor, _ in r):
                stats.n_saw_flop += 1
            stats._this_hand_actions = 0
            stats._this_hand_aggressive = 0
