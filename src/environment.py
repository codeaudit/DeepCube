#! /usr/bin/env python2
import numpy as np

from cube import Cube


class Environment(object):

    # Initialize a random cube
    def __init__(self, N, rand_nb):
        self.N = N
        self.cube = Cube(N=N)
        self.cube.randomize(rand_nb)

    # Make a move and get a reward:
    # 0 is the cube is not finish
    # 1 is the cube is done
    def perform_action(self, action):
        [f, l, d] = action
        self.cube.move(f, l, d)
        return self.cube.finish()

    # Select a random_action
    def random_action(self,):
        f = np.random.randint(6)
        l = np.random.randint(self.N)
        d = 1 + np.random.randint(3)
        return [f, l, d]

    def get_state(self,):
        return self.cube.stickers
