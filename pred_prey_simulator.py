import math
import sys
import ast
import numpy as np

from functools import partial
from sklearn.preprocessing import normalize
from prey_agent import PreyAgent
from nltk import Tree




def dfs(construct):
    if isinstance(construct, ast.Expr):
        dfs(construct.value)
    elif isinstance(construct, ast.Call):
        for arg in construct.args:
            dfs(arg)
        if construct.func.id == 'if_then_else':
            for call in filter(lambda x: isinstance(x, ast.Call), construct.args):
                call.args.insert(0, ast.Name(id=call.func.id))
                call.func.id = 'partial'


def partialise(expression):
    """Takes an expression and add `partial` for all calls when passed as args to `if_then_else`."""
    tree = ast.parse(expression)
    expr = tree.body[0]
    dfs(expr)
    return ast.unparse(tree)


class PredPreySimulator:

    def __init__(self, max_steps):
        # * Environment Properties *
        self.width = 200  # number of rows in discrete environment overlay, default 201
        self.height = 200  # number of columns in discrete environment overlay, default 201
        self.num_prey = 20  # number of prey, default 15
        # * Simulation Properties *
        self.max_moves = max_steps  # limit of steps per pred/prey per simulation loop
        self.max_speed = 1.5  # max speed of predator
        self.min_speed = 0.1  # min speed of predator
        self.vision_angle = 0.70710678  # radius in front of predator that can be seen
        self.vision_radius = 20  # distance in front of predator that can be seen
        self.sensing_radius = 5  # distance around predator that can be sensed for prey
        self.capture_radius = 1  # distance around the predator that allows predator to capture prey
        self.moves = 0  # number of steps that have been executed
        self.captured = 0  # number of prey captured by predator
        self.routine = None  # routine to be executed
        self.x_pos = self.width / 2  # x coordinate of predator in 2-d space
        self.y_pos = self.height / 2  # y coordinate of predator in 2-d space
        self.x_rot = 1  # x rotation of predator in 2-d space [-1, 1]
        self.y_rot = 0  # y rotation of predator in 2-d space [-1, 1]
        self.speed = 1  # speed of predator
        self.primitives = 0  # number of primitives in gp tree executed
        self.has_moved = False  # checks if predator has moved during execution
        # * Initialize Prey *
        self.prey = []
        # * Initialize Data *
        self.steps = []

    def run(self, individual):
        self.reset_environment()
        # working = "self.sin(self.if_then_else(partial(self.safe_div, self.prey_captured, 1.0), partial(self.safe_div, self.hit_wall, self.moves_remaining), partial(self.move_forward, 0.0)))"
        # original = "sin(if_then_else(safe_div(prey_captured, 1.0), safe_div(hit_wall, moves_remaining), move_forward(0.0)))"
        # original = "move_forward(if_then_else(add(0.0, sense_prey), average(moves_taken, moves_taken), if_then_else(-0.8623077206043339, prey_remaining, 1.0)))"
        # original = "less_than(if_then_else(moves_remaining, seek_prey, prey_remaining), average(prey_captured, seek_prey))"

        original = partialise(
            "less_than(if_then_else(moves_remaining, seek_prey, prey_remaining), average(prey_captured, seek_prey))")
        print("Original: " + str(original) + "\n")

        ind_tree = original.replace(", ", ")(")

        t = Tree.fromstring("(" + ind_tree + ")")
        t.pretty_print()

        function_list = ['add', 'sub', 'mul', 'safe_div', 'average', 'float_and', 'float_or', 'float_not',
                         'move_forward', 'rotate', 'if_then_else', 'greater_than', 'less_than', 'equal_to',
                         'sin', 'cos', 'seek_prey', 'sense_prey', 'hit_wall', 'prey_captured', 'prey_remaining',
                         'moves_taken', 'moves_remaining']

        for f in function_list:
            original = original.replace(f, "self." + f)

        print("Callable: " + original + "\n")

        try:
            eval(original)
        except MemoryError:
            _, _, traceback = sys.exc_info()
            raise MemoryError("Python cannot evaluate a tree higher than 90.")

    def reset_environment(self):
        # Predator Properties
        self.moves = 0  # number of steps that have been executed
        self.captured = 0  # number of prey captured by predator
        self.x_pos = self.width / 2  # x coordinate of predator in 2-d space
        self.y_pos = self.height / 2  # y coordinate of predator in 2-d space
        self.x_rot = 1
        self.y_rot = 0
        self.speed = 1  # speed of predator
        self.primitives = 0
        self.has_moved = False
        # * Initialize Prey *
        self.prey = [PreyAgent(self.width, self.height) for _ in range(self.num_prey)]
        # * Initialize Data *
        self.steps = []
        self.steps.append("PRED X = " + str(self.x_pos) + " Y = " + str(self.y_pos))  # predator starting position

    def print_prey_properties(self):
        for i, p in enumerate(self.prey):
            print("Prey " + str(i) + ": x_pos = " + str(round(p.x_pos, 2)) + ", y_pos = " + str(round(p.y_pos, 2)) +
                  ", x_rot = " + str(round(p.x_rot, 2)) + ", y_rot = " + str(round(p.y_rot, 2)))

    def print_pred_properties(self):
        print("Pred: x_pos = " + str(round(self.x_pos, 2)) + ", y_pos = " + str(round(self.y_pos, 2)) +
              ", x_rot = " + str(round(self.x_rot, 2)) + ", y_rot = " + str(round(self.y_rot, 2)))

    def if_then_else(self, condition, x, y):
        if callable(condition):
            condition = condition()
            print("IF_THEN_ELSE CONDITION = " + str(condition))
        if condition:
            if callable(x):
                x = x()
            return x
        else:
            if callable(y):
                y = y()
            return y

    def move_forward(self, x):
        print("move_forward")
        if callable(x):
            x = x()
        if self.moves < self.max_moves:
            self.moves += 1  # increase number of moves by 1
            self.has_moved = True  # proof predator has move_forward primitive in tree
            new_x_pos = self.x_pos + (self.x_rot * self.speed)  # calculate next x coordinate
            new_y_pos = self.y_pos + (self.y_rot * self.speed)  # calculate next y coordinate
            if 0 < new_x_pos < self.width and 0 < new_y_pos < self.height:
                self.x_pos = new_x_pos  # set pred to new x coordinate
                self.y_pos = new_y_pos  # set pred to new y coordinate
                self.steps.append("PRED X = " + str(self.x_pos) + " Y = " + str(self.y_pos))
            # * Prey Movement *
            for i, p in enumerate(self.prey):
                point1 = np.array((self.x_pos, self.y_pos))
                point2 = np.array((p.x_pos, p.y_pos))
                dist = np.linalg.norm(point1 - point2)
                if dist < self.capture_radius:
                    self.steps.append("CAPTURE X = " + str(self.x_pos) + " Y = " + str(self.y_pos))
                    self.captured += 1
                    del self.prey[i]  # delete prey from list after being captured
                p.move_forward()  # move the prey if not in capture radius of predator
        return x  # pass value through function without using it

    def rotate(self, x, y):
        print("rotate")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        # check if we need to truncate number to left of decimal
        if -1 <= x <= 1:
            new_x_rot = x
        else:
            new_x_rot = x - int(x)
        # check if we need to truncate number to left of decimal
        if -1 <= y <= 1:
            new_y_rot = y
        else:
            new_y_rot = y - int(y)
        # assign new rotation values
        self.x_rot = new_x_rot
        self.y_rot = new_y_rot
        # return average of rotation values as float
        return np.average(self.x_rot, self.y_rot)

    def set_speed(self, x):
        print("set_speed")
        if callable(x):
            x = x()
        # check if 0 < speed < 1 before assigning
        if x > 0:
            self.speed = min(x, self.max_speed)
        # return speed as float
        return self.speed

    def hit_wall(self):
        print("hit_wall")
        new_x_pos = self.x_pos + (self.x_rot * self.speed)  # calculate next x coordinate
        new_y_pos = self.y_pos + (self.y_rot * self.speed)  # calculate next y coordinate
        if not 0 < new_x_pos < self.width or not 0 < new_y_pos < self.height:
            return 1.0
        else:
            return 0.0

    def seek_prey(self):
        print("seek_prey")
        for p in self.prey:
            point1 = np.array((self.x_pos, self.y_pos))  # predator x, y position
            point2 = np.array((p.x_pos, p.y_pos))  # prey x, y position
            dist = np.sqrt(np.sum(np.square(point1 - point2)))  # euclidean distance
            # Check if prey is too far to be seen, then skip to next prey
            if dist > self.vision_radius:
                continue
            # Continue to check if prey is in sensing radius
            pred_rot = np.array((self.x_rot, self.y_rot))
            pred_rot_norm = normalize([pred_rot])
            pred_to_prey_rot = np.array((p.x_pos - self.x_pos, p.y_pos - self.y_pos))
            pred_to_prey_rot_norm = normalize([pred_to_prey_rot])
            dot_product = np.dot(pred_to_prey_rot_norm[0], pred_rot_norm[0])
            # Check if prey is in vision
            if dot_product >= self.vision_angle:
                return 1.0
        return 0.0

    def sense_prey(self):
        print("sense_prey")
        for p in self.prey:
            point1 = np.array((self.x_pos, self.y_pos))
            point2 = np.array((p.x_pos, p.y_pos))
            dist = np.linalg.norm(point1 - point2)
            if dist <= self.sensing_radius:
                return 1.0
        return 0.0

    def min_xy(self, x, y):
        print("min_xy")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        return np.min(x, y)

    def max_xy(self, x, y):
        print("max_xy")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        return np.max(x, y)

    def safe_div(self, x, y):
        print("safe_div")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        try:
            return x / y
        except ZeroDivisionError:
            return 1.0

    def average(self, x, y):
        print("average")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        return float(np.average(x, y))

    def float_and(self, x, y):
        print("float_and")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        if x and y:
            return 1.0
        else:
            return 0.0

    def float_or(self, x, y):
        print("float_or")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        if x or y:
            return 1.0
        else:
            return 0.0

    def float_not(self, x):
        print("float_not")
        if callable(x):
            x = x()
        if x:
            return 0.0
        else:
            return 1.0

    def greater_than(self, x, y):
        print("greater_than")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        if x > y:
            return 1.0
        else:
            return 0.0

    def less_than(self, x, y):
        print("less_than")
        print("x = " + str(x))
        print("y = " + str(y))
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        if x < y:
            return 1.0
        else:
            return 0.0

    def equal_to(self, x, y):
        print("equal_to")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        if x == y:
            return 1.0
        else:
            return 0.0

    def add(self, x, y):
        print("add")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        return x + y

    def sub(self, x, y):
        print("sub")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        return x - y

    def mul(self, x, y):
        print("mul")
        if callable(x):
            x = x()
        if callable(y):
            y = y()
        return x * y

    def sin(self, x):
        print("sin")
        if callable(x):
            x = x()
        return math.sin(x)

    def cos(self, x):
        print("cos")
        if callable(x):
            x = x()
        return math.cos(x)

    def prey_captured(self):
        print("prey_captured")
        return self.captured / self.num_prey

    def prey_remaining(self):
        print("prey_remaining")
        return (self.num_prey - self.captured) / self.num_prey

    def moves_taken(self):
        print("moves_taken")
        return self.moves / self.max_moves

    def moves_remaining(self):
        print("moves_remaining")
        return (self.max_moves - self.moves) / self.max_moves
