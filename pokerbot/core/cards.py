"""Card / Deck primitives.

Refactored out of Theory/probabilities.py so the rest of the package can
import without dragging in the CLI / equity calculator.
"""
from __future__ import annotations

from dataclasses import dataclass


VALID_RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
VALID_SUITS = ("H", "D", "C", "S")

RANK_TO_VALUE = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}


class InvalidCardError(ValueError):
    pass


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    def __post_init__(self) -> None:
        r = self.rank.upper() if len(self.rank) == 1 else self.rank
        if len(r) == 2 and r[0] == "1" and r[1] == "0":
            r = "10"
        s = self.suit.upper()
        if r not in VALID_RANKS or s not in VALID_SUITS:
            raise InvalidCardError(
                f"Invalid card: rank={self.rank!r}, suit={self.suit!r}"
            )
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
    """A simple 52-card deck you can remove known cards from."""

    def __init__(self) -> None:
        self.cards: list[Card] = [Card(r, s) for s in VALID_SUITS for r in VALID_RANKS]

    def remove(self, card: Card) -> None:
        try:
            self.cards.remove(card)
        except ValueError as exc:
            raise ValueError(f"Card not in deck: {card}") from exc

    def remove_many(self, cards) -> None:
        for c in cards:
            self.remove(c)

    def remaining(self) -> list[Card]:
        return list(self.cards)
