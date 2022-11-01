import copy
import random

import numpy as np

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


# TODO: Store moves/actions that predator takes
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

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x_pos = random.random() * (width - 1)  # x coordinate of prey in 2-d space
        self.y_pos = random.random() * (height - 1)  # y coordinate of prey in 2-d space
        self.x_rot = random.random() * 2 - 1  # x rotation of prey in 2-d space [-1, 1]
        self.y_rot = random.random() * 2 - 1  # y rotation of prey in 2-d space [-1, 1]
        self.speed = 0.7  # speed of prey

    def move_forward(self):
        new_x_pos = self.x_pos + (self.x_rot * self.speed)  # calculate next x coordinate
        new_y_pos = self.y_pos + (self.y_rot * self.speed)  # calculate next y coordinate
        if 0 < new_x_pos < self.width and 0 < new_y_pos < self.height:
            self.x_pos = new_x_pos  # set prey to new x coordinate
            self.y_pos = new_y_pos  # set prey to new y coordinate
        self.rotate()  # rotate in random direction before taking step forward

    def rotate(self):
        self.x_rot = random.random() * 2 - 1  # x rotation of prey in 2-d space [-1, 1]
        self.y_rot = random.random() * 2 - 1  # y rotation of prey in 2-d space [-1, 1]


class PredPreySimulator:

    def __init__(self, max_steps):
        # * Environment Properties *
        self.width = 200  # number of rows in discrete environment overlay, default 201
        self.height = 200  # number of columns in discrete environment overlay, default 201
        self.num_prey = 20  # number of prey, default 15
        # * Simulation Properties *
        self.max_moves = max_steps  # limit of steps per pred/prey per simulation loop
        self.moves = 0  # number of steps that have been executed
        self.captured = 0  # number of prey captured by predator
        self.routine = None
        # * Initialize Predator *
        self.x_pos = self.width / 2  # x coordinate of predator in 2-d space
        self.y_pos = self.height / 2  # y coordinate of predator in 2-d space
        self.x_rot = random.random() * 2 - 1  # x rotation of predator in 2-d space [-1, 1]
        self.y_rot = random.random() * 2 - 1  # y rotation of predator in 2-d space [-1, 1]
        self.speed = 1  # speed of predator
        self.max_speed = 1  # max speed of predator
        self.vision_angle = 90  # degrees in front of predator that can be seen
        self.vision_radius = 20  # distance in front of predator that can be seen
        # * Initialize Prey *
        self.prey = [PreyAgent(self.width, self.height) for _ in range(self.num_prey)]

    def print_prey_properties(self):
        for i, p in enumerate(self.prey):
            print("Prey " + str(i) + ": x_pos = " + str(round(p.x_pos, 2)) + ", y_pos = " + str(round(p.y_pos, 2)) +
                  ", x_rot = " + str(round(p.x_rot, 2)) + ", y_rot = " + str(round(p.y_rot, 2)))

    def print_pred_properties(self):
        print("Pred: x_pos = " + str(round(self.x_pos, 2)) + ", y_pos = " + str(round(self.y_pos, 2)) +
              ", x_rot = " + str(round(self.x_rot, 2)) + ", y_rot = " + str(round(self.y_rot, 2)))

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
            new_x_pos = self.x_pos + (self.x_rot * self.speed)  # calculate next x coordinate
            new_y_pos = self.y_pos + (self.y_rot * self.speed)  # calculate next y coordinate
            if 0 < new_x_pos < self.width and 0 < new_y_pos < self.height:
                self.x_pos = new_x_pos  # set pred to new x coordinate
                self.y_pos = new_y_pos  # set pred to new y coordinate
            # * Prey Movement *
            for i, p in enumerate(self.prey):
                p.move_forward()  # move the prey randomly

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
            self.speed = min(speed, self.max_speed)

    def seek_prey(self):
        for p in self.prey:
            u_x = p.x_pos - self.x_pos
            u_y = p.y_pos - self.y_pos
            e1_angle = None
            e2_angle = None

            if self.x_rot >= 0 and self.y_rot >= 0:
                x_rot = np.abs(self.x_rot)
                y_rot = np.abs(self.y_rot)
                e1_angle = np.arctan(y_rot / x_rot) + np.radians(self.vision_angle / 2)
                e2_angle = np.arctan(y_rot / x_rot) - np.radians(self.vision_angle / 2)
            elif self.x_rot < 0 and self.y_rot >= 0:
                x_rot = np.abs(self.x_rot)
                y_rot = np.abs(self.y_rot)
                e1_angle = np.arctan(y_rot / x_rot) + np.radians((self.vision_angle / 2) + np.radians(90))
                e2_angle = np.arctan(y_rot / x_rot) - np.radians((self.vision_angle / 2) + np.radians(90))
            elif self.x_rot < 0 and self.y_rot < 0:
                x_rot = np.abs(self.x_rot)
                y_rot = np.abs(self.y_rot)
                e1_angle = np.arctan(y_rot / x_rot) + np.radians((self.vision_angle / 2) + np.radians(180))
                e2_angle = np.arctan(y_rot / x_rot) - np.radians((self.vision_angle / 2) + np.radians(180))
            elif self.x_rot >= 0 and self.y_rot < 0:
                x_rot = np.abs(self.x_rot)
                y_rot = np.abs(self.y_rot)
                e1_angle = np.arctan(y_rot / x_rot) + np.radians((self.vision_angle / 2) + np.radians(270))
                e2_angle = np.arctan(y_rot / x_rot) - np.radians((self.vision_angle / 2) + np.radians(270))

            # calculate edge vectors of vision zone
            e1_x = self.vision_radius * np.cos(e1_angle) + self.x_pos
            e1_y = self.vision_radius * np.sin(e1_angle) + self.y_pos
            e2_x = self.vision_radius * np.cos(e2_angle) + self.x_pos
            e2_y = self.vision_radius * np.sin(e2_angle) + self.y_pos

            # calculate cross product between edge vectors and circle point to prey vector
            a = np.cross((e1_x, e1_y, 0), (u_x, u_y, 0))
            b = np.cross((u_x, u_y, 0), (e2_x, e2_y, 0))

            # check if position of prey is behind predator
            if np.dot((u_x, u_y), (self.x_rot, self.y_rot)) <= 0:
                continue
            # check if position of prey is beyond the viewing distance
            elif np.dot((u_x, u_y), (u_x, u_y)) > np.power(self.vision_radius, 2):
                continue
            # check if position of prey is within the viewing zone
            elif np.sign(a[2]) == np.sign(b[2]):
                print(str(np.sign(a[2])) + ", " + str(np.sign(b[2])))
                print("within vision " + str(p.x_pos) + ", " + str(p.y_pos))
                return True
            else:
                continue

    def sense_prey(self):
        for p in self.prey:
            point1 = np.array((self.x_pos, self.y_pos))
            point2 = np.array((p.x_pos, p.y_pos))
            dist = np.linalg.norm(point1 - point2)
            if dist < self.vision_radius:
                print("within radius = " + str(p.x_pos) + ", " + str(p.y_pos))
                return True


# Initialize simulation with 5000 steps
sim = PredPreySimulator(5000)
# sim.move_forward()
print("*** *** *** *** *** *** *** *** *** ***")
sim.print_pred_properties()
sim.print_prey_properties()
sim.seek_prey()
sim.sense_prey()


def main():
    random.seed(1)


if __name__ == "__main__":
    main()
