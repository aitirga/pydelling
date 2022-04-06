from __future__ import annotations

import numpy as np

from .BasePrimitive import BasePrimitive
from .Point import Point
import scipy.linalg as la
from typing import List

class Line(BasePrimitive):
    def __init__(self, p1: np.ndarray or Point or List = None, p2: np.ndarray or Point or List = None,
                 direction_vector: np.ndarray or List = None):
        if p1 is not None and p2 is not None:
            p1 = Point(p1)
            p2 = Point(p2)
            self.direction_vector = (p2 - p1) / np.linalg.norm(p2 - p1)
            if isinstance(p1, Point):
                self.p = p1
            else:
                self.p = Point(p1)

        elif direction_vector is not None and (p1 is not None or p2 is not None):
            self.direction_vector = direction_vector / np.linalg.norm(direction_vector)

            if p1 is not None:
                if isinstance(p1, Point):
                    self.p = p1
                else:
                    self.p = Point(p1)
            elif p2 is not None:
                if isinstance(p2, Point):
                    self.p = p2
                else:
                    self.p = Point(p2)

    def is_parallel(self, line: Line):
        return np.allclose(self.direction_vector, line.direction_vector)

    def angle(self, line: Line):
        return np.arccos(np.dot(self.direction_vector, line.direction_vector) /
                         (np.linalg.norm(self.direction_vector) * np.linalg.norm(line.direction_vector)))

    def intersect(self, primitive: BasePrimitive):
        if isinstance(primitive, Line):
            return self.intersect_line_line(primitive)

        else:
            raise NotImplementedError(f"Intersection with {type(primitive)} is not implemented")

    def intersect_line_line(self, line: Line):
        if self.is_parallel(line):
            return None
        else:
            delta_p = self.p - line.p
            a = np.array([
                [self.direction_vector[0], -line.direction_vector[0]],
                [self.direction_vector[1], -line.direction_vector[1]],
                [self.direction_vector[2], -line.direction_vector[2]]

            ])
            b = - np.array([
                [delta_p[0]],
                [delta_p[1]],
                [delta_p[2]],
            ])
            x = np.linalg.lstsq(a, b)
            if x[1] >= self.eps:
                return None
            else:
                return self.p + self.direction_vector * x[0][0]

    def __repr__(self):
        return f"Line(p:{self.p}, v:{self.direction_vector})"
