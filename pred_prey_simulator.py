import numpy as np

from sklearn.preprocessing import normalize

from functools import partial

from prey_agent import PreyAgent


def if_then_else(condition, out1, out2):
    out1() if condition() else out2()


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
        self.moves = None  # number of steps that have been executed
        self.captured = None  # number of prey captured by predator
        self.routine = None
        self.x_pos = None  # x coordinate of predator in 2-d space
        self.y_pos = None  # y coordinate of predator in 2-d space
        self.x_rot = None  # x rotation of predator in 2-d space [-1, 1]
        self.y_rot = None  # y rotation of predator in 2-d space [-1, 1]
        self.speed = None  # speed of predator
        # * Initialize Prey *
        self.prey = None
        # * Initialize Data *
        self.steps = None

    def run(self, routine):
        self.reset_environment()
        while self.moves < self.max_moves:
            routine()

    def reset_environment(self):
        # Predator Properties
        self.moves = 0  # number of steps that have been executed
        self.captured = 0  # number of prey captured by predator
        self.x_pos = self.width / 2  # x coordinate of predator in 2-d space
        self.y_pos = self.height / 2  # y coordinate of predator in 2-d space
        self.x_rot = 1
        self.y_rot = 0
        self.speed = 1  # speed of predator
        # * Initialize Prey *
        self.prey = [PreyAgent(self.width, self.height) for _ in range(self.num_prey)]
        # * Initialize Data *
        self.steps = []
        self.steps.append("PRED X = " + str(self.x_pos) + " Y = " + str(self.y_pos))

    def print_prey_properties(self):
        for i, p in enumerate(self.prey):
            print("Prey " + str(i) + ": x_pos = " + str(round(p.x_pos, 2)) + ", y_pos = " + str(round(p.y_pos, 2)) +
                  ", x_rot = " + str(round(p.x_rot, 2)) + ", y_rot = " + str(round(p.y_rot, 2)))

    def print_pred_properties(self):
        print("Pred: x_pos = " + str(round(self.x_pos, 2)) + ", y_pos = " + str(round(self.y_pos, 2)) +
              ", x_rot = " + str(round(self.x_rot, 2)) + ", y_rot = " + str(round(self.y_rot, 2)))

    def move_forward(self):
        if self.moves < self.max_moves:
            self.moves += 1  # increase number of moves by 1
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
                    del self.prey[i]
                p.move_forward()  # move the prey if not in capture radius of predator

    def increase_speed(self):
        if self.moves < self.max_moves:
            self.moves += 1
            if self.speed < self.max_speed:
                self.speed += 0.10

    def decrease_speed(self):
        if self.moves < self.max_moves:
            self.moves += 1
            if self.speed > self.min_speed:
                self.speed -= 0.10

    def turn_left(self):
        if self.moves < self.max_moves:
            self.moves += 1
            if self.x_rot >= 0 and self.y_rot >= 0:
                self.x_rot -= 0.10
                self.y_rot += 0.10
            elif self.x_rot < 0 and self.y_rot >= 0:
                self.x_rot -= 0.10
                self.y_rot -= 0.10
            elif self.x_rot < 0 and self.y_rot < 0:
                self.x_rot += 0.10
                self.y_rot -= 0.10
            elif self.x_rot >= 0 and self.y_rot < 0:
                self.x_rot += 0.10
                self.y_rot += 0.10

    def turn_right(self):
        if self.moves < self.max_moves:
            self.moves += 1
            if self.x_rot >= 0 and self.y_rot >= 0:
                self.x_rot += 0.10
                self.y_rot -= 0.10
            elif self.x_rot < 0 and self.y_rot >= 0:
                self.x_rot += 0.10
                self.y_rot += 0.10
            elif self.x_rot < 0 and self.y_rot < 0:
                self.x_rot -= 0.10
                self.y_rot += 0.10
            elif self.x_rot >= 0 and self.y_rot < 0:
                self.x_rot -= 0.10
                self.y_rot -= 0.10

    # deprecated
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

    # deprecated
    def set_speed(self, speed):
        # check if 0 < speed < 1 before assigning
        if speed > 0:
            self.speed = min(speed, self.max_speed)

    # deprecated
    def hit_wall(self):
        new_x_pos = self.x_pos + (self.x_rot * self.speed)  # calculate next x coordinate
        new_y_pos = self.y_pos + (self.y_rot * self.speed)  # calculate next y coordinate
        if not 0 < new_x_pos < self.width or not 0 < new_y_pos < self.height:
            return True
        else:
            return False

    # Checks if prey is in vision of predator
    def seek_prey(self):
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
                return True
            else:
                continue
        return False

    def if_seek_prey(self, out1, out2):
        return partial(if_then_else, self.seek_prey, out1, out2)

    # Checks if prey is in sensing radius of predator
    def sense_prey(self):
        for p in self.prey:
            point1 = np.array((self.x_pos, self.y_pos))
            point2 = np.array((p.x_pos, p.y_pos))
            dist = np.linalg.norm(point1 - point2)
            if dist <= self.sensing_radius:
                return True

    def if_sense_prey(self, out1, out2):
        return partial(if_then_else, self.sense_prey, out1, out2)

    # deprecated
    def moves(self):
        return self.moves

    # deprecated
    def captured(self):
        return self.captured

    # deprecated
    def x_pos(self):
        return self.x_pos

    # deprecated
    def y_pos(self):
        return self.y_pos

    # deprecated
    def x_rot(self):
        return self.x_rot

    # deprecated
    def y_rot(self):
        return self.y_rot