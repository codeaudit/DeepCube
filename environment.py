#! /usr/bin/env python2

from cube import cube

class Environment(Object):

    # Initialize a random cube
    def __init__(self, N, rand_nb):
        self.cube = Cube(N)
        self.cube.randomize(rand_nb)

    # Make a move and checks if the cube is ok
    def action(self, f, l, d):
        self.cube.move(f, l, d)
        return self.cube.finish()