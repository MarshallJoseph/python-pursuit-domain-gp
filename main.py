import copy
import random

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


class PredPreySimulator:

    def __init__(self, max_steps):
        # Simulation Properties
        self.max_moves = max_steps  # limit of steps per pred/prey per simulation loop
        self.moves = 0  # number of steps that have been executed
        self.captured = 0  # number of prey captured by predator
        # Predator Properties
        self.x_pos = 0
        self.y_pos = 0
        self.x_rot = 0
        self.y_rot = 0



def main():
    random.seed(1)


if __name__ == "__main__":
    main()
