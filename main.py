import math
import operator
import random

import deap.gp
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

# Initialize Primitive Set (ROOT nodes in format ("name", "numInputs"))
pset.addPrimitive(sim.add, 2)  # add
pset.addPrimitive(sim.sub, 2)  # subtract
pset.addPrimitive(sim.mul, 2)  # multiply
pset.addPrimitive(sim.safe_div, 2)  # divide
pset.addPrimitive(sim.average, 2)  # average
pset.addPrimitive(sim.float_and, 2)  # and gate
pset.addPrimitive(sim.float_or, 2)  # or gate
pset.addPrimitive(sim.float_not, 1)  # not gate
pset.addPrimitive(sim.move_forward, 1)  # move forward
pset.addPrimitive(sim.rotate, 2)  # rotate
pset.addPrimitive(sim.if_then_else, 3)
pset.addPrimitive(sim.greater_than, 2)  # x greater than y ?
pset.addPrimitive(sim.less_than, 2)  # x less than y ?
pset.addPrimitive(sim.equal_to, 2)  # x equal to y ?
pset.addPrimitive(sim.sin, 1)  # sin
pset.addPrimitive(sim.cos, 1)  # cos
# Initialize Terminal Set (Leaf nodes which hold a FLOAT value)
pset.addTerminal(sim.seek_prey)  # 1.0 if prey in vision radius, 0.0 if not
pset.addTerminal(sim.sense_prey)  # 1.0 if prey in sensing radius, 0.0 if not
pset.addTerminal(sim.hit_wall)  # 1.0 if next move hits wall, 0.0 if not
pset.addTerminal(sim.prey_captured)  # percent of prey captured
pset.addTerminal(sim.prey_remaining)  # percent of prey remaining
pset.addTerminal(sim.moves_taken)  # percent of moves taken
pset.addTerminal(sim.moves_remaining)  # percent of moves remaining
pset.addTerminal(1.0)  # boolean True converted to float
pset.addTerminal(0.0)  # boolean False converted to float
pset.addEphemeralConstant("ephemeral", lambda: random.uniform(-1, 1))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=2, max_=3)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#expr = gp.genFull(pset, min_=1, max_=3)
#tree_stuff = gp.PrimitiveTree(expr)
#print(str(tree_stuff))

tree = "sin(if_then_else(safe_div(prey_captured, 1.0), safe_div(hit_wall, moves_remaining), move_forward(0.0)))"



def eval_pred_prey(individual):
    # Run the generated individual
    sim.run(individual)
    #return sim.captured,


toolbox.register("evaluate", eval_pred_prey)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    random.seed(1)
    eval_pred_prey(tree)
#    pop = toolbox.population(n=100)
#    hof = tools.HallOfFame(1)
#    stats = tools.Statistics(lambda ind: ind.fitness.values)
#    stats.register("avg", np.mean)
#    stats.register("std", np.std)
#    stats.register("min", np.min)
#    stats.register("max", np.max)

#    algorithms.eaSimple(pop, toolbox, 0.9, 0.1, 1000, stats, halloffame=hof)

#    return pop, hof, stats


if __name__ == "__main__":
    main()
