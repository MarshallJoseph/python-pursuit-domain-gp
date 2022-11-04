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


# TODO: convert to strongly-typed gp


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


# deprecated
def max_xy(x, y):
    return max(x, y)


# deprecated
def min_xy(x, y):
    return min(x, y)


# deprecated
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


# Initialize simulation with 5000 steps
sim = PredPreySimulator(500)

primitive_set = gp.PrimitiveSet("MAIN", 0)

primitive_set.addPrimitive(sim.if_seek_prey, 2)
primitive_set.addPrimitive(sim.if_sense_prey, 2)
primitive_set.addPrimitive(prog2, 2)
primitive_set.addPrimitive(prog3, 3)

primitive_set.addTerminal(sim.move_forward)
primitive_set.addTerminal(sim.turn_left)
primitive_set.addTerminal(sim.turn_right)
primitive_set.addTerminal(sim.increase_speed)
primitive_set.addTerminal(sim.decrease_speed)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=primitive_set, min_=1, max_=2)

# Structure generator
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval_simulation(individual):
    # Transform the tree expression into functional Python code
    routine = gp.compile(individual, primitive_set)
    # Run the generated routine
    sim.run(routine)
    return sim.captured,


toolbox.register("evaluate", eval_simulation)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitive_set)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=11))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=11))


def main():
    random.seed(1)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.9, 0.1, 100, stats, halloffame=hof)

    return pop, hof, stats

if __name__ == "__main__":
    main()
