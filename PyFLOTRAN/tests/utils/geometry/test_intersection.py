import unittest
import numpy.testing as nptest
import numpy as np
from PyFLOTRAN.utils.geometry import Line
from PyFLOTRAN.preprocessing import DfnPreprocessor, MeshPreprocessor

class IntersectionCase(unittest.TestCase):
    def test_full_intersection(self):
        dfn_preprocessor = DfnPreprocessor()
        dfn_preprocessor.add_fracture(
            x=0.1, y=0.1, z=0.0, dip=0, dip_dir=0, size=1.2
        )
        mesh_preprocessor = MeshPreprocessor()
        mesh_preprocessor.add_tetrahedra(
            node_ids=np.array([0, 1, 2, 3]),
            node_coords=[
                np.array([0.0, 0.0, -0.5]),
                np.array([1.0, 0.0, -0.5]),
                np.array([0.0, 1.0, -0.5]),
                np.array([0.0, 0.0, 0.5]),
            ],
        )
        # Intersect fracture with mesh
        intersected_points = mesh_preprocessor.elements[0].intersect_faces_with_plane(
            dfn_preprocessor[0].plane)
        solution = [
            np.array([0.5, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.5, 0.0]),
        ]
        nptest.assert_array_almost_equal(intersected_points, solution)

    def test_full_intersection_rotated(self):
        dfn_preprocessor = DfnPreprocessor()
        dfn_preprocessor.add_fracture(
            x=0.1, y=0.1, z=0.0, dip=20, dip_dir=50, size=1.2
        )
        mesh_preprocessor = MeshPreprocessor()
        mesh_preprocessor.add_tetrahedra(
            node_ids=np.array([0, 1, 2, 3]),
            node_coords=[
                np.array([0.0, 0.0, -0.5]),
                np.array([1.0, 0.0, -0.5]),
                np.array([0.0, 1.0, -0.5]),
                np.array([0.0, 0.0, 0.5]),
            ],
        )
        # Intersect fracture with mesh
        intersected_points = mesh_preprocessor.elements[0].intersect_faces_with_plane(
            dfn_preprocessor[0].plane)
        solution = [
            np.array([0.62220399, 0.0, -0.12220399]),
            np.array([0.0, 0.0, 5.12772932e-02]),
            np.array([0.0, 5.85765892e-01, -8.57658923e-02]),
        ]
        nptest.assert_array_almost_equal(intersected_points, solution)

    def test_partial_intersection_1(self):

        dfn_preprocessor = DfnPreprocessor()
        dfn_preprocessor.add_fracture(
            x=0.1, y=0.1, z=0.5, dip=90, dip_dir=90, size=1.5
        )
        mesh_preprocessor = MeshPreprocessor()
        mesh_preprocessor.add_tetrahedra(
            node_ids=np.array([0, 1, 2, 3]),
            node_coords=[
                np.array([0.0, 0.0, -0.5]),
                np.array([1.0, 0.0, -0.5]),
                np.array([0.0, 1.0, -0.5]),
                np.array([0.0, 0.0, 0.5]),
            ],
        )
        # Intersect fracture with mesh
        intersected_points = mesh_preprocessor.elements[0].intersect_faces_with_plane(
            dfn_preprocessor[0].plane)

        solution = [
            np.array([0.1, 0.0, 0.4]),
            np.array([0.1, 0.0, -0.5]),
            np.array([0.1, 0.9, -0.5]),
        ]
        nptest.assert_array_almost_equal(intersected_points, solution)





if __name__ == "__main__":
    unittest.main()
