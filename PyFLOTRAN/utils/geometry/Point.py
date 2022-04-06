import numpy as np

from .BasePrimitive import BasePrimitive
from typing import *


class Point(BasePrimitive):
    def __init__(self, coords: np.ndarray or List):
        self.coords: np.ndarray = np.array(coords)

    def __repr__(self):
        return f"Point({self.coords})"
