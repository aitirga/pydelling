from __future__ import annotations

import logging
from typing import List

import numpy as np

from .Point import Point
from ._AbstractGidObject import _AbstractGidObject

logger = logging.getLogger(__name__)

class Line(_AbstractGidObject):
    local_id: int = 1
    global_id: int = 1
    lines: List = []
    has_copy = False
    def __init__(self, point_1: Point, point_2: Point):
        self.local_id = Line.local_id
        self.id = None
        if np.array_equal(point_1.coords, point_2.coords):
            raise ValueError('The start and end points of a line cannot be equal')
        self.point_1 = point_1
        self.point_2 = point_2
        self.points = [self.point_1, self.point_2]
        Line.local_id += 1

    def add(self):
        # Check if line already exists
        for line in Line.lines:
            if Line.check_lines_equal(self, line):
                self.id = line.id
                self.has_copy = True
                return ''

        self.id = Line.global_id
        Line.global_id += 1
        Line.lines.append(self)
        logger.info(f'Adding line {self.id} connecting {self.point_1} and {self.point_2}')
        export_str = 'Mescape Geometry Create Line\n'
        export_str += 'Join\n'
        export_str += f'{self.point_1.id}\n'
        export_str += f'{self.point_2.id}\n'
        return export_str

    @staticmethod
    def check_lines_equal(line_1: Line, line_2: Line):
        point_1_coords = [line_1.point_1.coords.tolist(), line_1.point_2.coords.tolist()]
        if line_2.point_1.coords.tolist() in point_1_coords and line_2.point_2.coords.tolist() in point_1_coords:
            return True
        else:
            return False

    def __repr__(self):
        if self.id:
            return f"Line {self.id}"
        else:
            return f"Line (local) {self.local_id}"







