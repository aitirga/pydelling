from typing import List

from .GidObject import GidObject
from .Line import Line
from .Point import Point


class Polyline(GidObject):
    """
    This class takes a list of points and creates a collection of lines
    """
    local_id: int = 1

    def __init__(self, points: List[Point], connect=False):
        super().__init__()
        self.lines = []
        self.points = points
        self.connect = connect
        self.set_up()

    def set_up(self):
        for idx, _ in enumerate(self.points[:-1]):
            point_a = self.points[idx]
            point_b = self.points[idx + 1]
            aux_line = Line(point_a, point_b)
            self.lines.append(aux_line)
        if self.connect:
            aux_line = Line(self.points[-1], self.points[0])
            self.lines.append(aux_line)

    def construct(self, *args, **kwargs):
        for point in self.points:
            self.add(point)
        for line in self.lines:
            self.add(line)