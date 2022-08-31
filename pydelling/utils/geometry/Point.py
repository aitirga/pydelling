from __future__ import annotations

import numpy as np


class Point(np.ndarray):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        if len(input_array) == 1:
            raise ValueError("Point must have 2 or 3 coordinates")
        elif len(input_array) == 2:
            obj = np.asarray(input_array).view(cls)
        elif len(input_array) == 3:
            obj = np.asarray(input_array).view(cls)
        else:
            raise ValueError("Point must have 2 or 3 coordinates")
        return obj

    def distance(self, p: Point):
        """Computes euclidean distance between two points"""
        diff = self - p
        return float(np.sqrt((diff ** 2).sum()))

    def __repr__(self):
        return f"Point({self})"


    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    def get_json(self):
        return [self.x, self.y, self.z]
