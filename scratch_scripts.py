# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:49:51 2018

@author: David
"""
from math import sqrt
from random import randint

class Rocket():
    # Rocket simulates a rocket ship for a game or a physics simulations
    
    def __init__(self, x=0, y=0):
        # Each rock has an (x,y) position
        self.x = x
        self.y = y
    
    def move_rocket(self, x_increment=0, y_increment=1):
        # Increment the y-position of the rocket
        self.y += y_increment
        self.x += x_increment
        
    def get_distance(self, other_rocket):
        # Calculates distance from this rocket and another rocket, and returns the distance
        distance = sqrt((self.x - other_rocket.x)**2 + (self.y - other_rocket.y)**2)
        return distance
    
class Shuttle(Rocket):
    # Shuttle is really just a reuasable rocket.
    
    def __init__(self, x=0, y=0, flights_completed=0):
        super().__init__(x,y)
        self.flights_completed = flights_completed

# Create several shuttles with random positions and random numbers of flights completed.
shuttles = []
for x in range(0,3):
    x = randint(0,100)
    y = randint(1,100)
    flights_completed = randint(0,10)
    shuttles.append(Shuttle(x,y,flights_completed))

rockets = []
for x in range(0,3):
    x = randint(0,100)
    y = randint(1,100)
    rockets.append(Rocket(x,y))

# Show the number of filgths completed for each shuttle
for idx, shuttle in enumerate(shuttles):
    print("Shuttle %d has completed %d flights." % (idx, shuttle.flights_completed))

print('\n')
# Show the distance from the first shuttle to all other shuttles
first_shuttle = shuttles[0]
for idx, shuttle in enumerate(shuttles):
    distance = first_shuttle.get_distance(shuttle)
    print("The first shuttle is %f units away from shuttle %d" %(distance, idx))

print('\n')
# Show the distance from the first shuttle to all other rockets
for idx, rocket in enumerate(rockets):
    distance = first_shuttle.get_distance(rocket)
    print("The first shuttle is %f units away from rocket %d." % (distance, idx))