import numpy as np

from pydelling.preprocessing.mesh_preprocessor.geometry import BaseFace


class TriangleFace(BaseFace):
    def __init__(self, node_ids, node_coords):
        super().__init__(node_ids, node_coords)
        self.type = "triangle"

    def compute_centroid(self):
        return np.mean(self.coords, axis=0)

    @property
    def edges(self):
        return [
            [self.nodes[0], self.nodes[1]],
            [self.nodes[1], self.nodes[2]],
            [self.nodes[2], self.nodes[0]]
        ]

    @property
    def edge_vectors(self):
        return [
            self.coords[1] - self.coords[0],
            self.coords[2] - self.coords[1],
            self.coords[0] - self.coords[2]
        ]
