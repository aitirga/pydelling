import numpy as np

from pydelling.readers.iGPReader.geometry import BaseFace


class QuadrilateralFace(BaseFace):
    def __init__(self, nodes, coords):
        super().__init__(nodes, coords)
        self.type = "Quadrilateral"

    def compute_centroid(self):
        return np.mean(self.coords, axis=0)
        # t1 = [0, 1, 3]
        # t2 = [1, 2, 3]
        # triangles = [t1, t2]
        # poly_centroid = np.zeros(shape=3)
        # for triangle_nodes in triangles:
        #     # Set-up variables
        #     q1 = self.node_coords[triangle_nodes[0]]  # Point1 of small triangle
        #     q2 = self.node_coords[triangle_nodes[1]]  # Point2 of small triangle
        #     q3 = self.node_coords[triangle_nodes[2]]  # Point3 of small triangle
        #     q_v = np.array([q1, q2, q3])
        #     v1 = q2 - q1
        #     v2 = q3 - q1
        #     v_cross = np.cross(v1, v2)
        #     area_triangle = np.linalg.norm(v_cross) / 2.0
        #     # print(f"area triangle:{area_triangle}")
        #     # print(f"Total area:{self.area}")
        #     # Compute centroid
        #     mean_centroid = np.mean(q_v, axis=0)  # Centroid of small triangle
        #     poly_centroid += area_triangle * mean_centroid
        #     # print(f"q1: {q1}, q2: {q2}, q3:{q3}, mean_centroid:{mean_centroid}")
        # return poly_centroid / self.area
