"""7-card hand evaluator + showdown helpers.

This is the slow-but-correct reference implementation. For training-time
speed we'll later add a precomputed lookup, but the public API stays the same.
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

from pokerbot.core.cards import Card, RANK_TO_VALUE


@dataclass
class Player:
    seat: int
    hole: tuple[Card, Card]


# Hand category constants — higher beats lower.
CAT_HIGH = 0
CAT_PAIR = 1
CAT_TWO_PAIR = 2
CAT_TRIPS = 3
CAT_STRAIGHT = 4
CAT_FLUSH = 5
CAT_FULL_HOUSE = 6
CAT_QUADS = 7
CAT_STRAIGHT_FLUSH = 8


class HandEvaluator:
    """Hand ranking + showdown helpers."""

    @classmethod
    def showdown(
        cls, players: list[Player], board: list[Card]
    ) -> tuple[list[int], list[tuple]]:
        if len(board) != 5:
            raise ValueError("showdown() expects exactly 5 board cards")
        if not players:
            raise ValueError("showdown() needs at least one player")

        scores = [cls.score_seven(list(p.hole) + board) for p in players]
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        return winners, scores

    @staticmethod
    def win_shares(num_players: int, winner_indexes: list[int]) -> list[float]:
        if num_players <= 0:
            raise ValueError("num_players must be positive")
        if not winner_indexes:
            raise ValueError("winner_indexes cannot be empty")
        share = 1.0 / len(winner_indexes)
        out = [0.0] * num_players
        for i in winner_indexes:
            out[i] = share
        return out

    @classmethod
    def score_seven(cls, cards: list[Card]) -> tuple:
        if len(cards) != 7:
            raise ValueError("score_seven() expects exactly 7 cards")
        best_score: tuple | None = None
        for five in itertools.combinations(cards, 5):
            score = cls.score_five(list(five))
            if best_score is None or score > best_score:
                best_score = score
        if best_score is None:
            raise RuntimeError("No 5-card combinations generated")
        return best_score

    @staticmethod
    def _straight_high(ranks_desc: list[int]) -> int | None:
        uniq = sorted(set(ranks_desc), reverse=True)
        if len(uniq) != 5:
            return None
        if uniq == [14, 5, 4, 3, 2]:  # wheel
            return 5
        if uniq[0] - uniq[4] == 4:
            return uniq[0]
        return None

    @classmethod
    def score_five(cls, cards: list[Card]) -> tuple:
        if len(cards) != 5:
            raise ValueError("score_five() expects exactly 5 cards")

        ranks = sorted([RANK_TO_VALUE[c.rank] for c in cards], reverse=True)
        suits = [c.suit for c in cards]

        is_flush = len(set(suits)) == 1
        straight_high = cls._straight_high(ranks)

        c = Counter(ranks)
        counts = sorted(c.values(), reverse=True)

        if straight_high is not None and is_flush:
            return (CAT_STRAIGHT_FLUSH, straight_high)

        if counts == [4, 1]:
            quad_rank = max(r for r, cnt in c.items() if cnt == 4)
            kicker = max(r for r, cnt in c.items() if cnt == 1)
            return (CAT_QUADS, quad_rank, kicker)

        if counts == [3, 2]:
            trips_rank = max(r for r, cnt in c.items() if cnt == 3)
            pair_rank = max(r for r, cnt in c.items() if cnt == 2)
            return (CAT_FULL_HOUSE, trips_rank, pair_rank)

        if is_flush:
            return (CAT_FLUSH, *ranks)

        if straight_high is not None:
            return (CAT_STRAIGHT, straight_high)

        if counts == [3, 1, 1]:
            trips_rank = max(r for r, cnt in c.items() if cnt == 3)
            kickers = sorted((r for r, cnt in c.items() if cnt == 1), reverse=True)
            return (CAT_TRIPS, trips_rank, *kickers)

        if counts == [2, 2, 1]:
            pairs = sorted((r for r, cnt in c.items() if cnt == 2), reverse=True)
            kicker = max(r for r, cnt in c.items() if cnt == 1)
            return (CAT_TWO_PAIR, pairs[0], pairs[1], kicker)

        if counts == [2, 1, 1, 1]:
            pair_rank = max(r for r, cnt in c.items() if cnt == 2)
            kickers = sorted((r for r, cnt in c.items() if cnt == 1), reverse=True)
            return (CAT_PAIR, pair_rank, *kickers)

        return (CAT_HIGH, *ranks)
