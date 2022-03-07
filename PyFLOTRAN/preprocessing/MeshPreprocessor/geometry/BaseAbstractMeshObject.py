import numpy as np
from typing import *

class BaseAbstractMeshObject:
    '''Class for the base abstract mesh object. Provides method and attribute namespace'''
    nodes: np.ndarray  # Node id set
    coords: np.ndarray   # Coordinates of each node
    local_id = int  # Element id
    type: str or None
    meshio_type: str or None