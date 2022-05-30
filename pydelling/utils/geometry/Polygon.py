from .BasePrimitive import BasePrimitive
from . import Point, Segment
import numpy as np
from typing import List
from pydelling.utils.geometry_utils import order_points_clockwise


class Polygon(BasePrimitive):
    def __init__(self, points: List[Point] or np.ndarray):
        # The set of points should be ordered in a clockwise fashion
        self.points = order_points_clockwise(points)
        self.segments = self.generate_segments()

    def generate_segments(self):
        segments = []
        for i in range(len(self.points)):
            segments.append(Segment(self.points[i], self.points[(i + 1) % len(self.points)]))
        return segments

    def to_csv(self, filename='polygon.csv'):
        import csv
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for point in self.points:
                writer.writerow(point)

                



