from __future__ import annotations

from typing import List

import numpy as np

from .BasePrimitive import BasePrimitive
from .Point import Point


class Vector(np.ndarray, BasePrimitive):
    coords: np.ndarray
    def __new__(cls, v: np.ndarray or List=None, p1: Point or List or np.ndarray=None, p2: Point or List or np.ndarray=None, *args, **kwargs):
        if v is not None:
            assert len(v) >= 2, "Vector must have 2 or 3 coordinates"
            if isinstance(v, Vector):
                obj = np.asarray(v).view(cls)
            else:
                obj = np.asarray(v).view(cls)
            return obj


        elif p1 is not None and p2 is not None:
            p1 = Point(p1)
            p2 = Point(p2)
            obj = np.asarray(p2 - p1).view(cls)
            return obj

        else:
            raise ValueError("Vector must be initialized with either a list or a numpy array")

    def __repr__(self):
        return f"Vector({self})"

