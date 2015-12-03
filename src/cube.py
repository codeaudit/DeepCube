#!/usr/bin/env python2

"""
usage
-----
- initialize a solved cube with `c = Cube(N)` where `N` is the side length.
- randomize a cube with `c.randomize(32)` where `32` is the number of random moves to make.
- make cube moves with `c.move()` and turn the whole cube with `c.turn()`.
- make figures with `c.render().savefig(fn)` where `fn` is the filename.
- change sticker colors with, eg, `c.stickercolors[c.colordict["w"]] = "k"`.
conventions
-----------
- This is a model of where the stickers are, not where the solid cubies are.  That's a bug not a feature.
- Cubes are NxNxN in size.
- The faces have integers and one-letter names. The one-letter face names are given by the dictionary `Cube.facedict`.
- The layers of the cube have names that are composed of a face letter and a number, with 0 indicating the outermost face.
- Every layer has two layer names, for instance, (F, 1) and (B, 1) are the same layer of a 3x3x3 cube; (F, 1) and (B, 3) are the same layer of a 5x5x5.
- The colors have integers and one-letter names. The one-letter color names are given by the dictionary `Cube.colordict`.
- Convention is x before y in face arrays, plus an annoying baked-in left-handedness.  Sue me.  Or fork, fix, pull-request.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon


class Cube(object):
    """
    Cube
    ----
    Initialize with arguments:
    - `N`, the side length (the cube is `N`x`N`x`N`)
    - optional `whiteplastic=True` if you like white cubes
    """
    facedict = {"U": 0, "D": 1, "F": 2, "B": 3, "R": 4, "L": 5}
    inv_facedict = {0, "U", 1: "D", 2: "F", 3: "B", 4: "R", 5: "L"}
    dictface = dict([(v, k) for k, v in facedict.items()])

    def __init__(self, N):
        self.N = N
        self.stickers = np.array(
            [np.tile(i, (self.N, self.N)) for i in range(6)])

    def move(self, i, l, d):
        """
        Make a layer move of layer `l` parallel to face `inv_facedict[f]` through
        `d` 90-degree turns in the clockwise direction.  Layer `0` is
        the face itself, and higher `l` values are for layers deeper
        into the cube.  Use `d=3` or `d=-1` for counter-clockwise
        moves, and `d=2` for a 180-degree move..
        i is in range(6)
        l is in range(N)
        d is in range(1, 4)
        """
        f = inv_facedict[i]
        l2 = self.N - 1 - l
        assert l < self.N
        ds = range((d + 4) % 4)
        if f == "U":
            f2 = "D"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["F"], range(self.N), l2),
                              (self.facedict["R"], range(self.N), l2),
                              (self.facedict["B"], range(self.N), l2),
                              (self.facedict["L"], range(self.N), l2)])
        if f == "D":
            return self.move("U", l2, -d)
        if f == "F":
            f2 = "B"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["U"], range(self.N), l),
                              (self.facedict["L"], l2, range(self.N)),
                              (self.facedict["D"], range(self.N)[::-1], l2),
                              (self.facedict["R"], l, range(self.N)[::-1])])
        if f == "B":
            return self.move("F", l2, -d)
        if f == "R":
            f2 = "L"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["U"], l2, range(self.N)),
                              (self.facedict["F"], l2, range(self.N)),
                              (self.facedict["D"], l2, range(self.N)),
                              (self.facedict["B"], l, range(self.N)[::-1])])
        if f == "L":
            return self.move("R", l2, -d)
        for d in ds:
            if l == 0:
                self.stickers[i] = np.rot90(self.stickers[i], 3)
            if l == self.N - 1:
                self.stickers[i2] = np.rot90(self.stickers[i2], 1)
        return None

    def _rotate(self, args):
        """
        Internal function for the `move()` function.
        """
        a0 = args[0]
        foo = self.stickers[a0]
        a = a0
        for b in args[1:]:
            self.stickers[a] = self.stickers[b]
            a = b
        self.stickers[a] = foo
        return None

    def randomize(self, number):
        """
        Make `number` randomly chosen moves to scramble the cube.
        """
        for _ in range(number):
            f = self.dictface[np.random.randint(6)]
            l = np.random.randint(self.N)
            d = 1 + np.random.randint(3)
            self.move(f, l, d)
        return None

    def finish(self,):
        return np.array_equal(self.stickers, np.array([np.tile(i, (self.N, self.N)) for i in range(6)]))


# TESTING
if __name__ == "__main__":
    np.random.seed(42)
    c = Cube(6)
    c.move("U", 0, -1)
    c.move("U", 0, 1)
    c.randomize(5)
    c.finish()
