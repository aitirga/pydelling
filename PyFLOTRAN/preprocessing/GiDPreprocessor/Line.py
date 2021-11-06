import numpy as np
from ._AbstractGidObject import _AbstractGidObject
from functools import wraps
from .Point import Point
import logging

logger = logging.getLogger(__file__)


class Line(_AbstractGidObject):
    local_id: int = 1
    global_id: int = 1
    def __init__(self, point_1: Point, point_2: Point):
        self.local_id = Line.local_id
        self.id = None
        self.point_1 = point_1
        self.point_2 = point_2
        Line.local_id += 1

    def add(self):
        self.id = Line.global_id
        logger.info(f'Adding line {self.id} connecting {self.point_1} and {self.point_2}')
        Line.global_id += 1
        export_str = 'Mescape Geometry Create Line\n'
        export_str += 'Join\n'
        export_str += f'{self.point_1.id}\n'
        export_str += f'{self.point_2.id}\n'
        return export_str

    def __repr__(self):
        return f"Line {self.local_id}"


