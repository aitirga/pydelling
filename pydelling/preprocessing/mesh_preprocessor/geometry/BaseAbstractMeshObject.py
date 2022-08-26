import numpy as np


class BaseAbstractMeshObject:
    '''Class for the base abstract mesh object. Provides method and attribute namespace'''
    nodes: np.ndarray  # Node id set
    coords: np.ndarray   # Coordinates of each node
    local_id = int  # Element id
    type: str or None
    meshio_type: str or None
    centroid: np.ndarray  # Centroid of the element

    @property
    def edges(self):
        '''Returns the edges of the element'''
        return None

    @property
    def edge_vectors(self):
        '''Returns the edge vectors of the element'''
        return None

