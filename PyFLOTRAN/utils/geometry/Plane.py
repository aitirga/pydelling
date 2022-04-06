from __future__ import annotations
from .BasePrimitive import BasePrimitive
from .Point import Point
from .Line import Line
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

    def intersect(self, primitive: BasePrimitive):
        """Returns the intersection of this plane with the given primitive"""
        if isinstance(primitive, Plane):
            return self._intersect_plane_plane(plane=primitive)
        else:
            raise NotImplementedError(f"Intersection with {type(primitive)} is not implemented")

    def is_parallel(self, plane: Plane):
        """Returns True if this plane is parallel to the given plane"""
        if np.isclose(np.dot(self.n.coords, plane.n.coords), 1):
            return True

    def _intersect_plane_plane(self, plane: Plane):
        """Performs the intersection of this plane with the given plane"""
        if self.is_parallel(plane):
            return None
        normal_a = self.n.coords
        normal_b = plane.n.coords
        U = np.cross(normal_a, normal_b)
        M = np.array((normal_a, normal_b, U))
        X = np.array((-normal_a, -normal_b, np.zeros_like(normal_a)))
        # print(M)
        p_inter = np.linalg.solve(M, X).T
        p1 = p_inter[0]
        p2 = (p_inter + U)[0]
        intersected_line = Line(p1, p2)

        return intersected_line