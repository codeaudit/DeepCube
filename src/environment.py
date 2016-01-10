#! /usr/bin/env python2
import numpy as np

from cube import Cube


class Environment(object):

    # Initialize a random cube
    def __init__(self, N):
        self.N = N
        self.cube = Cube(N=N)

    def suffle(self, rand_nb=None, fixed_action=None):
        if rand_nb is not None:
            moves = self.cube.randomize(rand_nb)
            return moves
        self.perform_action(fixed_action)
        return fixed_action

    # Make a move and get a reward:
    # 0 is the cube is not finish
    # 1 is the cube is done
    def perform_action(self, action):
        [f, l, d] = action
        self.cube.move(f, l, d)
        return self.reward()

    def reward(self,):
        return self.cube.finish()

        # if self.cube.finish():
        #     return 1.
        # for i in range(6):
        #     if np.array_equal(self.cube.stickers[i, :, :], i * np.ones((self.N, self.N))):
        #         return  0.1
        # return 0.

    # Select a random_action
    def random_action(self,):
        f = np.random.randint(6)
        l = np.random.randint(self.N)
        d = 1 + np.random.randint(3)
        return [f, l, d]

    def get_state(self,):
        return self.cube.stickers
