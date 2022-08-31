import logging
from typing import List

import numpy as np

from ._AbstractGidObject import _AbstractGidObject

logger = logging.getLogger(__name__)

class Point(_AbstractGidObject):
    has_copy = False
    local_id: int = 1
    global_id: int = 1
    points: List = []
    def __init__(self, coords):
        self.local_id = Point.local_id
        self.id = None
        Point.local_id += 1
        if isinstance(coords, list):
            coords = np.array(coords)
            assert coords.shape[0] == 3, 'A 3D coordinate needs to be given'
        assert isinstance(coords, np.ndarray), 'Please provide a numpy array or a list'
        self.coords = coords

    def add(self):
        for point in Point.points:
            if np.array_equal(point.coords, self.coords):
                self.id = point.id
                self.has_copy = True
                return ''

        self.id = Point.global_id
        Point.global_id += 1
        Point.points.append(self)
        logger.info(f'Adding point {self.id} at {self.coords}')
        export_str = 'Mescape Geometry Create Point\n'
        export_str += f'{",".join(map(str, self.coords))}\n'
        return export_str

    @property
    def coords_comma(self):
        return f'{",".join(map(str, self.coords))}'

    def __repr__(self):
        if self.id:
            return f"Point {self.id} ({self.coords})"
        else:
            return f"Point (local) {self.local_id} ({self.coords})"



