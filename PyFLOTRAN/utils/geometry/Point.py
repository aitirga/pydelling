import numpy as np

from .BasePrimitive import BasePrimitive
from typing import *


class Point(BasePrimitive):
    def __init__(self, coords: np.ndarray or List):
        if len(coords) == 1:
            raise ValueError("Point must have 2 or 3 coordinates")
        elif len(coords) == 2:
            self.coords = np.array([coords[0], coords[1], 0])
        elif len(coords) == 3:
            self.coords = np.array(coords)
        else:
            raise ValueError("Point must have 2 or 3 coordinates")

    def __repr__(self):
        return f"Point({self.coords})"

    def __str__(self):
        return f"Point({self.coords})"

    def __eq__(self, other):
        if isinstance(other, Point):
            return np.all(self.coords == other.coords)
        else:
            return np.all(self.coords == other)

    def __add__(self, other):
        if isinstance(other, Point):
            return self.coords + other.coords
        else:
            return self.coords + other

    def __sub__(self, other):
        if isinstance(other, Point):
            return self.coords - other.coords
        else:
            return self.coords - other

    def __mul__(self, other):
        return self.coords * other

    def __len__(self):
        return len(self.coords)

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def z(self):
        return self.coords[2]


