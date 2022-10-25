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


# TODO: Create movement for predator/prey and test their functionality
# TODO: Create rotation for predator/prey and test their functionality
# TODO: Create sensing function for predator
# TODO: Create speed update function for predator and test their functionality
# TODO: Add terminals/operators for GP tree
# TODO: Test GP tree generation with DEAP


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


class PreyAgent:

    def __init__(self, rows, cols):
        self.x_pos = random.random() * (rows - 1)  # x coordinate of prey in 2-d space
        self.y_pos = random.random() * (cols - 1)  # y coordinate of prey in 2-d space
        self.x_rot = random.random() * 2 - 1  # x rotation of prey in 2-d space [-1, 1]
        self.y_rot = random.random() * 2 - 1  # y rotation of prey in 2-d space [-1, 1]
        self.speed = 0.7  # speed of prey

    #  def move_forward(self):


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
        self.x_pos = self.rows / 2  # x coordinate of predator in 2-d space
        self.y_pos = self.cols / 2  # y coordinate of predator in 2-d space
        self.x_rot = random.random() * 2 - 1  # x rotation of predator in 2-d space [-1, 1]
        self.y_rot = random.random() * 2 - 1  # y rotation of predator in 2-d space [-1, 1]
        self.speed = 1  # speed of predator
        # Initialize Discrete Environment (OPEN, PRED, PREY, DEAD)
        self.matrix = [['OPEN' for _ in range(self.cols)] for _ in range(self.rows)]  # fill matrix with 'OPEN' cells
        # Initialize Predator
        self.matrix[int(self.y_pos)][int(self.x_pos)] = 'PRED'  # initialize predator location on discrete overlay
        print("predator " + "x: " + str(self.x_pos) + ", y: " + str(self.y_pos))
        # Initialize Prey
        self.prey = [PreyAgent(self.rows, self.cols) for _ in range(self.num_prey)]
        for i, p in enumerate(self.prey):
            px_pos = int(p.x_pos)  # remove decimals for discrete overlay
            py_pos = int(p.y_pos)  # remove decimals for discrete overlay
            print("prey " + str(i) + ": x: " + str(px_pos) + ", y: " + str(py_pos))
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

    def move_forward(self):
        if self.moves < self.max_moves:
            self.moves += 1  # increase number of moves by 1
            # reset current cell in discrete overlay since we are going to move the pred
            self.matrix[int(self.y_pos)][int(self.x_pos)] = 'OPEN'
            next_x_pos = self.x_pos + (self.x_rot * self.speed)  # calculate next x coordinate
            next_y_pos = self.y_pos + (self.y_rot * self.speed)  # calculate next y coordinate
            # set new cell where pred is located to 'PRED'
            self.matrix[int(next_y_pos)][int(next_x_pos)] = 'PRED'
            self.x_pos = next_x_pos  # set pred to new x coordinate
            self.y_pos = next_y_pos  # set pred to new y coordinate

    def rotate(self, new_x_rot, new_y_rot):
        # check conditions of new x rotation
        if new_x_rot > 1:
            new_x_rot = 1
        elif new_x_rot < -1:
            new_x_rot = -1
        # check conditions of new y rotation
        if new_y_rot > 1:
            new_y_rot = 1
        elif new_y_rot < -1:
            new_y_rot = -1
        # assign new x and y rotations
        self.x_rot = new_x_rot
        self.y_rot = new_y_rot

    def set_speed(self, speed):
        # check if 0 < speed < 1 before assigning
        if speed > 0:
            self.speed = min(speed, 1)


# Initialize simulation with 5000 steps
sim = PredPreySimulator(5000)


def main():
    random.seed(1)


if __name__ == "__main__":
    main()
