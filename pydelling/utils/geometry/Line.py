from __future__ import annotations

from typing import List

import numpy as np

from .BasePrimitive import BasePrimitive
from .Point import Point


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
        if np.isclose(np.dot(self.direction_vector, line.direction_vector), 1):
            return True
        if np.isclose(np.dot(self.direction_vector, line.direction_vector), -1):
            return True

    def angle(self, line: Line):
        return np.arccos(np.dot(self.direction_vector, line.direction_vector) /
                         (np.linalg.norm(self.direction_vector) * np.linalg.norm(line.direction_vector)))

    def intersect(self, primitive: BasePrimitive):
        from .intersections import intersect_line_line, intersect_plane_line

        if primitive.__class__.__name__ == "Line":
            return intersect_line_line(line_1=self, line_2=primitive)
        elif primitive.__class__.__name__ == "Plane":
            return intersect_plane_line(line=self, plane=primitive)



        else:
            raise NotImplementedError(f"Intersection with {type(primitive)} is not implemented")


    def __repr__(self):
        return f"Line(point: {self.p}, direction_vector: {self.direction_vector})"

    def __str__(self):
        return f"Line(point: {self.p}, direction_vector: {self.direction_vector})"
