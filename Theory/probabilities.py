# file to teach myself how to calculate the probability of winning given a hand. Every hand is seen, at last at first

from __future__ import annotations
import itertools
import math
import random
from dataclasses import dataclass # Exact enumeration of every board completion; above this, use Monte Carlo (still only randomness in runouts).

_MAX_EXACT_RUNOUTS = 800_000
_MC_TRIALS = 400_000

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
    '''The 52-card deck that is used in the game.'''
    @staticmethod
    def full_deck() -> list[Card]:
        return [Card(r, s) for s in VALID_SUITS for r in VALID_RANKS]


@dataclass
class Player:
    seat: int
    hole: tuple[Card, Card]


class HandEvaluator:
    """Hand ranking + showdown helpers."""

    @classmethod
    def showdown(cls, players: list[Player], board: list[Card]) -> tuple[list[int], list[tuple]]:
        """
        Return (winner_indexes, scores) for a completed 5-card board.

        winner_indexes contains indexes into `players` (0-based).
        """
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
        """
        Convert winners into per-seat win share (sums to 1.0).
        """
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
        """
        Score a 7-card hand by taking best score across all 5-card subsets.
        Higher tuple must mean stronger hand.
        """
        if len(cards) != 7:
            raise ValueError("score_seven() expects exactly 7 cards")

        # Evaluate all C(7,5)=21 subsets and keep the best score.
        best_score: tuple | None = None
        for five in itertools.combinations(cards, 5):
            score = cls.score_five(list(five))
            if best_score is None or score > best_score:
                best_score = score
        if best_score is None:
            raise RuntimeError("No 5-card combinations generated")
        return best_score


    @classmethod
    def score_five(cls, cards: list[Card]) -> tuple:
        """
        Return comparable score tuple for exactly 5 cards.

        Suggested format:
        (category, tiebreak1, tiebreak2, ...)
        where larger tuples are always stronger.
        """
        if len(cards) != 5:
            raise ValueError("score_five() expects exactly 5 cards")

        # You implement hand category + kicker math here.
        raise NotImplementedError("Implement score_five()")


def prompt_player_count() -> int:
    while True:
        try:
            player_num = int(input("Enter the number of players (2-10): ").strip())
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if 2 <= player_num <= 10:
            return player_num
        print("Invalid number of players. Please enter a number between 2 and 10.")


def prompt_cards_known() -> str:
    while True:
        cards_known = input("Do you know the other players' cards? (y/n): ").strip().lower()
        if cards_known in {"y", "n"}:
            return cards_known
        print("Invalid input. Please enter 'y' or 'n'.")

def prompt_known_players(player_num: int) -> list[Player]:
    print("Enter 2 cards per player (e.g. '2H 3H').")
    used_cards: set[Card] = set()
    players: list[Player] = []

    for i in range(player_num):
        while True:
            cards_input = input(f"Enter the cards for player {i + 1}: ")
            try:
                cards = parse_cards_line(cards_input)
            except InvalidCardError as err:
                print(err)
                continue

            if len(cards) != 2:
                print("Please enter exactly 2 cards.")
                continue
            if cards[0] == cards[1]:
                print("A player cannot have duplicate hole cards.")
                continue
            if cards[0] in used_cards or cards[1] in used_cards:
                print("Those cards are already used by another player.")
                continue

            used_cards.add(cards[0])
            used_cards.add(cards[1])
            players.append(Player(seat=i + 1, hole=(cards[0], cards[1])))
            break

    return players


def run_cli() -> None:
    player_num = prompt_player_count()
    print(f"Starting the program for {player_num} players...")

    cards_known = prompt_cards_known()
    if cards_known == "y":
        players = prompt_known_players(player_num)
        print("Cards captured successfully:")
        for p in players:
            print(f"Player {p.seat}: {p.hole[0]} {p.hole[1]}")
    else:
        # TODO: implement Monte Carlo path when opponents' cards are unknown
        pass


if __name__ == "__main__":
    run_cli()
    