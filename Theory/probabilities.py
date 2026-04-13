# file to teach myself how to calculate the probability of winning given a hand. Every hand is seen, at last at first


import random

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
        


