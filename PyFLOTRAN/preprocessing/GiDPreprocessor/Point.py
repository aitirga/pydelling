import numpy as np
from ._AbstractGidObject import _AbstractGidObject
from functools import wraps
import logging

logger = logging.getLogger(__file__)

class Point(_AbstractGidObject):
    local_id: int = 1
    global_id: int = 1
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
        self.id = Point.global_id
        logger.info(f'Adding point {self.id} at {self.coords}')
        Point.global_id += 1
        export_str = 'Mescape Geometry Create Point\n'
        export_str += f'{",".join(map(str, self.coords))}\n'
        return export_str

    def __repr__(self):
        return f"Point {self.local_id} ({self.coords})"


