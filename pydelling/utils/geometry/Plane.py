from __future__ import annotations

from typing import List

import numpy as np

from .BasePrimitive import BasePrimitive
from .Point import Point
from .Vector import Vector


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
        from .intersections import intersect_plane_plane, intersect_plane_line, intersect_plane_segment
        if primitive.__class__.__name__ == "Plane":
            return intersect_plane_plane(plane_1=self, plane_2=primitive)
        elif primitive.__class__.__name__ == "Line":
            return intersect_plane_line(plane=self, line=primitive)
        elif primitive.__class__.__name__ == 'Segment':
            return intersect_plane_segment(plane=self, segment=primitive)

        else:
            raise NotImplementedError(f"Intersection with {type(primitive)} is not implemented")

    def is_parallel(self, plane: Plane):
        """Returns True if this plane is parallel to the given plane"""
        if np.isclose(np.dot(self.n, plane.n), 1):
            return True
        if np.isclose(np.dot(self.n, plane.n), -1):
            return True

