"""Configurable heuristic Leduc players.

Each player is parameterized by a behavioral profile. Instead of hand-coding
strategies, we describe *why* the player would act:

    - bet_strong_freq:    P(bet/raise | I hold a strong hand for the spot)
    - bluff_freq:         P(bluff | I hold air)
    - call_strong_freq:   P(call | facing bet, I have a strong hand)
    - call_marginal_freq: P(call | facing bet, I have a marginal hand)
    - call_weak_freq:     P(call | facing bet, I have garbage)
    - tilt_susceptibility: how much the profile shifts after a big loss

"Strong/marginal/weak" is interpreted in context:
    Round 1:  K = strong, Q = marginal, J = weak
    Round 2:  pair (matches board) = strong; high card not paired = marginal;
              low card not paired = weak.

Tilt mechanic: after losing >5 chips in the last 3 hands, an opponent's
profile is multiplicatively perturbed:
    - call_weak_freq        *= (1 + 2.0 * tilt_susceptibility)
    - bluff_freq            *= (1 + 1.5 * tilt_susceptibility)
    - call_marginal_freq    *= (1 + 1.0 * tilt_susceptibility)
    - bet_strong_freq stays the same (good hands are always bet)

This produces realistic tilt behavior without needing full mood modeling.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from pokerbot.games.leduc import LeducPoker, card_rank


# -- Strength categorization for Leduc --

def hand_strength(hole_card: int, board_card: int | None) -> str:
    """Return 'strong' / 'marginal' / 'weak'."""
    rh = card_rank(hole_card)
    if board_card is None:
        # Round 1: K=strong, Q=marginal, J=weak
        return ["weak", "marginal", "strong"][rh]
    rb = card_rank(board_card)
    if rh == rb:
        return "strong"     # made a pair
    if rh > rb:
        # Higher card than the board (an "overcard"): marginal
        return "marginal"
    return "weak"           # under the board


# -- Profiles --

@dataclass
class PlayerProfile:
    name: str
    bet_strong_freq: float
    bluff_freq: float
    call_strong_freq: float
    call_marginal_freq: float
    call_weak_freq: float
    raise_when_betting: float = 0.30   # if betting/raising is legal, prob to raise vs bet
    tilt_susceptibility: float = 0.0


PROFILES: dict[str, PlayerProfile] = {
    "tight_passive": PlayerProfile(
        name="tight_passive",
        bet_strong_freq=0.85, bluff_freq=0.05,
        call_strong_freq=0.95, call_marginal_freq=0.30, call_weak_freq=0.05,
        raise_when_betting=0.20, tilt_susceptibility=0.1,
    ),
    "loose_aggressive": PlayerProfile(
        name="loose_aggressive",
        bet_strong_freq=0.95, bluff_freq=0.40,
        call_strong_freq=0.95, call_marginal_freq=0.65, call_weak_freq=0.25,
        raise_when_betting=0.55, tilt_susceptibility=0.2,
    ),
    "calling_station": PlayerProfile(
        name="calling_station",
        bet_strong_freq=0.60, bluff_freq=0.05,
        call_strong_freq=1.0, call_marginal_freq=0.85, call_weak_freq=0.55,
        raise_when_betting=0.05, tilt_susceptibility=0.1,
    ),
    "maniac": PlayerProfile(
        name="maniac",
        bet_strong_freq=1.0, bluff_freq=0.65,
        call_strong_freq=1.0, call_marginal_freq=0.85, call_weak_freq=0.50,
        raise_when_betting=0.75, tilt_susceptibility=0.4,
    ),
    "tilt_prone": PlayerProfile(
        name="tilt_prone",
        bet_strong_freq=0.85, bluff_freq=0.20,
        call_strong_freq=0.95, call_marginal_freq=0.50, call_weak_freq=0.15,
        raise_when_betting=0.30,
        tilt_susceptibility=0.7,    # drops a gear when losing
    ),
}


def make_player(profile_name: str, rng: random.Random | None = None) -> "LeducHeuristicPlayer":
    return LeducHeuristicPlayer(PROFILES[profile_name], rng=rng)


# -- The player --

class LeducHeuristicPlayer:
    """Decides Leduc actions based on hand strength + a behavioral profile.

    Maintains its own short memory of recent results so that tilt can kick in.
    """

    def __init__(self, profile: PlayerProfile, rng: random.Random | None = None):
        self.profile = profile
        self.rng = rng or random.Random()
        self._recent_results: deque = deque(maxlen=5)

    # --- public ---

    def observe_result(self, net_chips: float) -> None:
        self._recent_results.append(net_chips)

    def is_tilted(self) -> bool:
        """A simple ground-truth tilt signal: lost >5 chips in the recent window."""
        if len(self._recent_results) < 3:
            return False
        return sum(self._recent_results) < -5.0

    def effective_profile(self) -> PlayerProfile:
        """Apply tilt perturbation if we're tilted."""
        if not self.is_tilted():
            return self.profile
        s = self.profile.tilt_susceptibility
        return PlayerProfile(
            name=self.profile.name + "*tilted",
            bet_strong_freq=self.profile.bet_strong_freq,
            bluff_freq=min(1.0, self.profile.bluff_freq * (1 + 1.5 * s)),
            call_strong_freq=self.profile.call_strong_freq,
            call_marginal_freq=min(1.0, self.profile.call_marginal_freq * (1 + 1.0 * s)),
            call_weak_freq=min(1.0, self.profile.call_weak_freq * (1 + 2.0 * s)),
            raise_when_betting=self.profile.raise_when_betting,
            tilt_susceptibility=s,
        )

    def decide(
        self,
        game: LeducPoker,
        state,
        legal_actions: list[str],
        my_player: int,
    ) -> str:
        """Pick one action."""
        cards, history = state
        hole = cards[my_player]
        board = cards[2] if len(cards) >= 3 else None
        strength = hand_strength(hole, board)
        prof = self.effective_profile()

        # Unwind the situation: are we facing a bet, or first-to-act?
        round_history = history.split("/")[-1] if "/" in history else history

        if not round_history or round_history[-1] == "k":
            # First action of the round, or our opponent checked. Decide bet vs check.
            if strength == "strong":
                bet_p = prof.bet_strong_freq
            elif strength == "marginal":
                # Marginal: bet sometimes for value/bluff
                bet_p = 0.5 * prof.bet_strong_freq + 0.3 * prof.bluff_freq
            else:
                bet_p = prof.bluff_freq
            if "b" in legal_actions and self.rng.random() < bet_p:
                return "b"
            if "k" in legal_actions:
                return "k"
            # Fallback: take the first legal action
            return legal_actions[0]

        # Facing a bet/raise: decide call/fold/raise
        if strength == "strong":
            call_p = prof.call_strong_freq
        elif strength == "marginal":
            call_p = prof.call_marginal_freq
        else:
            call_p = prof.call_weak_freq

        # Decide raise first (only if legal AND we'd have called anyway)
        if "r" in legal_actions and self.rng.random() < call_p:
            if self.rng.random() < prof.raise_when_betting:
                return "r"

        if self.rng.random() < call_p:
            if "c" in legal_actions:
                return "c"
        if "f" in legal_actions:
            return "f"
        # Shouldn't happen — fallback
        return legal_actions[0]
