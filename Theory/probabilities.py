# file to teach myself how to calculate the probability of winning given a hand. Every hand is seen, at last at first

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

VALID_RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
VALID_SUITS = ("H", "D", "C", "S")

RANK_TO_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

_SUIT_IDS = {"H": 0, "D": 1, "C": 2, "S": 3}
_INDEX_COMBOS_7_5 = tuple(itertools.combinations(range(7), 5))


class InvalidCardError(ValueError):
    pass

# @dataclass is a decorator that creates a class with __init__ and __repr__ methods. Just easier to use.
@dataclass(frozen=True)
class Card:
    """A card in a deck of 52 cards."""

    rank: str
    suit: str
    
    def __post_init__(self) -> None:
        # in @dataclass, __post_init__ is a method that is called after the object is initialized.
        r = self.rank.upper() if len(self.rank) == 1 else self.rank
        if len(r) == 2 and r[0] == "1" and r[1] == "0":
            r = "10"
        s = self.suit.upper()
        if r not in VALID_RANKS or s not in VALID_SUITS:
            raise InvalidCardError(f"Invalid card: rank={self.rank!r}, suit={self.suit!r}")
        object.__setattr__(self, "rank", r)
        object.__setattr__(self, "suit", s)

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


def parse_card(token: str) -> Card:
    t = token.strip().upper()
    if not t:
        raise InvalidCardError("Empty card token")
    if t.startswith("10"):
        if len(t) < 3:
            raise InvalidCardError(f"Invalid card: {token!r}")
        return Card("10", t[2])
    return Card(t[0], t[1])


def parse_cards_line(line: str) -> list[Card]:
    parts = line.strip().split()
    return [parse_card(p) for p in parts]


class Deck:
    @staticmethod
    def full_deck() -> list[Card]:
        return [Card(r, s) for s in VALID_SUITS for r in VALID_RANKS]


@dataclass
class Player:
    seat: int
    hole: tuple[Card, Card]


class HandEvaluator:
    pass