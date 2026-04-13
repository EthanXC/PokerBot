# file to teach myself how to calculate the probability of winning given a hand. Every hand is seen, at last at first


import random

'''
I want to design this to be object oriented.
'''

print("Game Start")

while True:
    players = int(input("Player amount:"))
    if players == isinstance(True, int) and players >= 2 and players <= 10:
        break
    else:
        print("Please enter an integer between 2 and 10")

