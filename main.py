import copy
import random

import numpy as np

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from array import *

# terminals in the gp will execute as the routine runs, this will cause the predator
# agent to move and rotate. each time the predator moves, one move will be consumed.

# TURN RIGHT
# if 0 < y < 1 && -1 < x < 0, then increase y, increase x
# if 0 < y < 1 && 0 < x < 1, then decrease y, increase x
# if -1 < y < 0 && 0 < x < 1, then decrease y, decrease x
# if -1 < y < 0 && -1 < x < 0, then increase y, decrease x

# TURN LEFT
# if 0 < y < 1 && -1 < x < 0, then decrease y, decrease x
# if 0 < y < 1 && 0 < x < 1, then increase y, decrease x
# if -1 < y < 0 && 0 < x < 1, then increase y, increase x
# if -1 < y < 0 && -1 < x < 0, then decrease y, increase x


class PreyAgent:

    def __init__(self, rows, cols):
        self.x_pos = random.random() * (rows - 1)  # x coordinate of prey in 2-d space
        self.y_pos = random.random() * (cols - 1)  # y coordinate of prey in 2-d space
        self.x_rot = random.random() * 2 - 1  # x rotation of prey in 2-d space [-1, 1]
        self.y_rot = random.random() * 2 - 1  # y rotation of prey in 2-d space [-1, 1]
        self.speed = 0.7  # speed of prey


class PredPreySimulator:

    def __init__(self, max_steps):
        # Environment Properties
        self.rows = 21  # number of rows in discrete environment overlay, default 201
        self.cols = 21  # number of columns in discrete environment overlay, default 201
        self.num_prey = 8  # number of prey, default 15
        # Simulation Properties
        self.max_moves = max_steps  # limit of steps per pred/prey per simulation loop
        self.moves = 0  # number of steps that have been executed
        self.captured = 0  # number of prey captured by predator
        self.routine = None
        # Predator Properties
        self.x_pos = int(self.rows / 2)  # x coordinate of predator in 2-d space
        self.y_pos = int(self.cols / 2)  # y coordinate of predator in 2-d space
        self.x_rot = random.random() * 2 - 1  # x rotation of predator in 2-d space [-1, 1]
        self.y_rot = random.random() * 2 - 1  # y rotation of predator in 2-d space [-1, 1]
        self.speed = 1  # speed of predator
        # Initialize Prey
        self.prey = [PreyAgent(self.rows, self.cols) for _ in range(self.num_prey)]
        # Initialize Discrete Environment (OPEN, PRED, PREY, DEAD)
        self.matrix = [['OPEN' for _ in range(self.cols)] for _ in range(self.rows)]  # fill matrix with 'OPEN' cells
        self.matrix[self.y_pos][self.x_pos] = 'PRED'  # initialize predator location on discrete overlay
        for p in self.prey:
            px_pos = int(p.x_pos)  # remove decimals for discrete overlay
            py_pos = int(p.y_pos)  # remove decimals for discrete overlay
            print("x: " + str(px_pos) + ", y: " + str(py_pos))
            self.matrix[py_pos][px_pos] = 'PREY'  # initialize prey location on discrete overlay

        for row in self.matrix:  # print discrete overlay in matrix form
            print(row)

    def reset_environment(self):
        # Predator Properties
        self.x_pos = 100  # x coordinate of predator in 2-d space
        self.y_pos = 100  # y coordinate of predator in 2-d space
        self.x_rot = random.random() * 2 - 1  # x rotation of predator in 2-d space [-1, 1]
        self.y_rot = random.random() * 2 - 1  # y rotation of predator in 2-d space [-1, 1]
        self.speed = 1  # speed of predator


# Initialize simulation with 5000 steps
sim = PredPreySimulator(5000)


def main():
    random.seed(1)


if __name__ == "__main__":
    main()
