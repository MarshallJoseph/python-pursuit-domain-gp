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





sim = PredPreySimulator(5000)  # Initialize simulation with 5000 steps

pset = gp.PrimitiveSet("MAIN", 0)  # Initialize primitive set called "MAIN" with 0 inputs

pset.addPrimitive(sim.add, 2)  # add
pset.addPrimitive(sim.sub, 2)  # subtract
pset.addPrimitive(sim.mul, 2)  # multiply
pset.addPrimitive(sim.safe_div, 2)  # divide
pset.addPrimitive(sim.average, 2)  # average
pset.addPrimitive(sim.float_and, 2)  # and gate
pset.addPrimitive(sim.float_or, 2)  # or gate
pset.addPrimitive(sim.float_not, 1)  # not gate
pset.addPrimitive(sim.greater_than, 2)  # x greater than y ?
pset.addPrimitive(sim.less_than, 2)  # x less than y ?
pset.addPrimitive(sim.equal_to, 2)  # x equal to y ?
pset.addPrimitive(sim.sin, 1)  # sin
pset.addPrimitive(sim.cos, 1)  # cos

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
