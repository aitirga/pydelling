import numpy as np

from pydelling.readers.iGPReader.geometry import BaseFace


class TriangleFace(BaseFace):
    def __init__(self, nodes, coords):
        super().__init__(nodes, coords)
        self.type = "Triangle"

    def compute_centroid(self):
        return np.mean(self.coords, axis=0)
