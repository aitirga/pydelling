from .BasePrimitive import BasePrimitive
from .Point import Point
from typing import List
import numpy as np


class Vector(BasePrimitive):
    def __init__(self, v: np.ndarray or List=None, p1: Point or List or np.ndarray=None, p2: Point or List or np.ndarray=None):
        if v is not None:
            assert len(v) >= 2, "Vector must have 2 or 3 coordinates"
            self.coords = np.array(v)
        elif p1 is not None and p2 is not None:
            p1 = Point(p1)
            p2 = Point(p2)
            self.coords = p2 - p1

    def __repr__(self):
        return f"Vector({self.coords})"

    def __str__(self):
        return f"Vector({self.coords})"

    def __len__(self):
        return len(self.coords)