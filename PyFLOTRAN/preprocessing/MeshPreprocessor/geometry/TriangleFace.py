import numpy as np

from PyFLOTRAN.preprocessing.MeshPreprocessor.geometry import BaseFace


class TriangleFace(BaseFace):
    def __init__(self, node_ids, node_coords):
        super().__init__(node_ids, node_coords)
        self.type = "triangle"

    def compute_centroid(self):
        return np.mean(self.coords, axis=0)
