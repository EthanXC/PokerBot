"""Card abstraction for tractable CFR on bigger games.

Maps the 169 distinct preflop hand classes into N equity buckets so CFR
can share strategy across hands of similar strength.
"""

from pokerbot.abstraction.preflop_buckets import (
    NUM_BUCKETS,
    hand_class_str,
    bucket_for_hand,
    all_169_classes,
    cached_class_equities,
    cached_bucket_map,
    cached_bucket_equity_matrix,
)
from pokerbot.abstraction.postflop_buckets import (
    NUM_POSTFLOP_BUCKETS,
    bucket_for_strength,
    bucket_for_hole_and_board,
    bucket_name,
    cached_postflop_bucket_equities,
)

__all__ = [
    "NUM_BUCKETS",
    "hand_class_str",
    "bucket_for_hand",
    "all_169_classes",
    "cached_class_equities",
    "cached_bucket_map",
    "cached_bucket_equity_matrix",
    "NUM_POSTFLOP_BUCKETS",
    "bucket_for_strength",
    "bucket_for_hole_and_board",
    "bucket_name",
    "cached_postflop_bucket_equities",
]
