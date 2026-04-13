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

# @dataclass is a decorator that creates a class with __init__ and __repr__ methods. Just easier to use.
@dataclass(frozen=True)

class Card:
    rank: str
    suit: str
    '''
    A card in a deck of 52 cards.
    '''
    pass



class Deck:
    pass

class HandEvaluator:
    pass

class EquitySimulator:
    '''
    Calculates the equity of a hand against a range of hands. Or, a simulated probability of winning when you
    cannot know the other hands.
    '''
    pass