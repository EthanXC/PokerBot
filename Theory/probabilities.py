# file to teach myself how to calculate the probability of winning given a hand. Every hand is seen, at last at first


import random
from dataclasses import dataclass

'''
I want to design this to be object oriented.
'''

print("Game Start")

while True:
    try:
        players = int(input("Player amount:"))
    except ValueError:
        players = None # marks invalid input

    if players is not None and 2 <= players <= 10:
        break

    print ("Please enter an integer between 2 and 10")

VALID_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
VALID_SUITS = ['H', 'D', 'C', 'S']

class InvalidCardError(ValueError):
    pass

# @dataclass is a decorator that creates a class with __init__ and __repr__ methods. Just easier to use.
@dataclass(frozen=True)
class Card:
    '''
    A card in a deck of 52 cards.
    '''

    rank: str
    suit: str
    
    def __post_init__(self):
        # in @dataclass, __post_init__ is a method that is called after the object is initialized.
        r = self.rank.upper() if len(self.rank) == 1 else self.rank
        s = self.suit.lower()

        # normalize if you use single letters
        if r not in VALID_RANKS or s not in VALID_SUITS:
            raise InvalidCardError(f"Invalid card: rank={self.rank!r}, suit={self.suit!r}")



class Deck:
    pass

@dataclass
class Player:
    seat: int
    hole: tuple[Card, Card] | None = None # or list, filled after dealing
    stack: int = 10000
    chips: int = 0
    is_dealer: bool = False
    is_small_blind: bool = False
    is_big_blind: bool = False
    is_button: bool = False
    is_small_blind: bool = False
    is_big_blind: bool = False
    is_button: bool = False
    is_active: bool = True

    pass

class HandEvaluator:
    pass

class EquitySimulator:
    '''
    Calculates the equity of a hand against a range of hands. Or, a simulated probability of winning when you
    cannot know the other hands.
    '''
    pass