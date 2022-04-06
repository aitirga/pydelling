from .BasePrimitive import BasePrimitive
from .Point import Point
from .Vector import Vector
import numpy as np
from typing import List


class Plane(BasePrimitive):
    def __init__(self, point: Point or List or np.ndarray, normal: List or Vector or np.ndarray):
        self.p = Point(point)
        self.n = Vector(normal)

    def __repr__(self):
        return f"Plane(point:{self.p}, normal:{self.n})"

    def __str__(self):
        return f"Plane(point:{self.p}, normal:{self.n})"

