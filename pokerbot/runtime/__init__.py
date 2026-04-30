"""Runtime subpackage — playing actual hands and harvesting training data.

Pieces:
  - heuristic_player.py: configurable synthetic Leduc players (tight, loose,
                         bluff-heavy, calling-station, tilted variants).
  - session.py:          run N hands between two policies, record full traces.
  - features.py:         turn raw traces into (bluff features, label) and
                         (tilt features, label) tensors. Labels come from
                         showdowns where we observe the cards.
  - data_generation.py:  high-level helpers that produce ready-to-train
                         datasets for the classifiers.

The point: the bluff and tilt classifiers should learn from REAL game traces
where the labels are determined by actual cards at showdown, not from features
I made up.
"""

from pokerbot.runtime.heuristic_player import (
    LeducHeuristicPlayer,
    PROFILES,
    make_player,
)
from pokerbot.runtime.session import HandTrace, run_session
from pokerbot.runtime.features import (
    extract_bluff_examples,
    extract_tilt_examples,
)
from pokerbot.runtime.data_generation import (
    build_bluff_dataset,
    build_tilt_dataset,
)
from pokerbot.runtime.match import (
    StrategyPlayer,
    MatchResult,
    play_match,
)
from pokerbot.runtime.adaptive_bot import AdaptiveBotPlayer
from pokerbot.runtime.nlhe_player import (
    NLHEHeuristicPlayer,
    PROFILES as NLHE_PROFILES,
    make_nlhe_player,
    preflop_strength,
    postflop_strength,
)
from pokerbot.runtime.nlhe_match import play_table, TableMatchResult
from pokerbot.runtime.multi_opponent_bot import MultiOpponentAdaptiveBot

__all__ = [
    "LeducHeuristicPlayer",
    "PROFILES",
    "make_player",
    "HandTrace",
    "run_session",
    "extract_bluff_examples",
    "extract_tilt_examples",
    "build_bluff_dataset",
    "build_tilt_dataset",
    "StrategyPlayer",
    "MatchResult",
    "play_match",
    "AdaptiveBotPlayer",
    "NLHEHeuristicPlayer",
    "NLHE_PROFILES",
    "make_nlhe_player",
    "preflop_strength",
    "postflop_strength",
    "play_table",
    "TableMatchResult",
    "MultiOpponentAdaptiveBot",
]
