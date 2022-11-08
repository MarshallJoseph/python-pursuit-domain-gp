import operator
import random

import numpy as np

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from pred_prey_simulator import PredPreySimulator


def min_xy(x, y):
    return np.min(x, y)


def max_xy(x, y):
    return np.max(x, y)


def safe_div(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 1


def average(x, y):
    return np.average(x, y)


sim = PredPreySimulator(500)  # Initialize simulation with 5000 steps

pset = gp.PrimitiveSet("MAIN", 0)  # Initialize primitive set called "MAIN" with 0 inputs

pset.addPrimitive(operator.add, 2)  # add
pset.addPrimitive(operator.sub, 2)  # subtract
pset.addPrimitive(operator.mul, 2)  # multiply
pset.addPrimitive(safe_div, 2)  # divide
pset.addPrimitive(average, 2)



def main():
    random.seed(1)


if __name__ == "__main__":
    main()
