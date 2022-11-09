import math
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


def float_and(x, y):
    if x and y:
        return 1
    else:
        return 0


def float_or(x, y):
    if x or y:
        return 1
    else:
        return 0


def float_not(x):
    if x:
        return 0
    else:
        return 1


def greater_than(x, y):
    if x > y:
        return 1
    else:
        return 0


def less_than(x, y):
    if x < y:
        return 1
    else:
        return 0


def equal_to(x, y):
    if x == y:
        return 1
    else:
        return 0


sim = PredPreySimulator(5000)  # Initialize simulation with 5000 steps

pset = gp.PrimitiveSet("MAIN", 0)  # Initialize primitive set called "MAIN" with 0 inputs

pset.addPrimitive(operator.add, 2)  # add
pset.addPrimitive(operator.sub, 2)  # subtract
pset.addPrimitive(operator.mul, 2)  # multiply
pset.addPrimitive(safe_div, 2)  # divide
pset.addPrimitive(average, 2)  # average
pset.addPrimitive(float_and, 2)  # and gate
pset.addPrimitive(float_or, 2)  # or gate
pset.addPrimitive(float_not, 1)  # not gate
pset.addPrimitive(greater_than, 2)  # x greater than y ?
pset.addPrimitive(less_than, 2)  # x less than y ?
pset.addPrimitive(equal_to, 2)  # x equal to y ?
pset.addPrimitive(math.sin, 1)  # sin
pset.addPrimitive(math.cos, 1)  # cos

pset.addTerminal(sim.seek_prey)  # 1 if prey in vision radius, 0 if not
pset.addTerminal(sim.sense_prey)  # 1 if prey in sensing radius, 0 if not
pset.addTerminal(sim.hit_wall)  # 1 if next move hits wall, 0 if not
pset.addTerminal(sim.prey_captured)  # percent of prey captured
pset.addTerminal(sim.prey_remaining)  # percent of prey remaining
pset.addTerminal(sim.moves_taken)  # percent of moves taken
pset.addTerminal(sim.moves_remaining)  # percent of moves remaining
pset.addTerminal(1)  # boolean True converted to float
pset.addTerminal(0)  # boolean False converted to float
pset.addEphemeralConstant("ephemeral", lambda: random.uniform(-1, 1))


def main():
    random.seed(1)


if __name__ == "__main__":
    main()
