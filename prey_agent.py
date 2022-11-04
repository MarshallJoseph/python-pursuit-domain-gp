import random


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
        self.rotate()

    def rotate(self):
        self.x_rot = random.random() * 2 - 1  # x rotation of prey in 2-d space [-1, 1]
        self.y_rot = random.random() * 2 - 1  # y rotation of prey in 2-d space [-1, 1]