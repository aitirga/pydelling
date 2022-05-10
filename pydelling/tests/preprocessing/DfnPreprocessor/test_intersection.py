import unittest
import numpy.testing as nptest
import numpy as np
from pydelling.utils.geometry import Line
from pydelling.preprocessing import DfnPreprocessor, MeshPreprocessor
from pydelling.preprocessing.dfn_preprocessor import DfnUpscaler

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


    def test_partial_intersection_2(self):

        dfn_preprocessor = DfnPreprocessor()
        dfn_preprocessor.add_fracture(
            x=0.1, y=0.1, z=0.5, dip=75, dip_dir=85, size=1.5
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
            np.array([0.14876171, 0.0, 0.35123829]),
            np.array([0.37772158, 0.0, -0.5]),
            np.array([0.31805952, 0.68194048, -0.5]),
        ]
        nptest.assert_array_almost_equal(intersected_points, solution)


    def test_partial_intersection_3(self):
        dfn_preprocessor = DfnPreprocessor()
        dfn_preprocessor.add_fracture(
            x=0.1, y=0.1, z=0.3, dip=75, dip_dir=65, size=1.3
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
            np.array([0.12422918, 0.0, 0.37577082]),
            np.array([0.38315014, 0.0, -0.5]),
            np.array([0.0, 5.12725439e-01, -1.27254385e-02]),
            np.array([0.0, 0.82166813, -0.5]),
        ]
        nptest.assert_array_almost_equal(intersected_points, solution)


    def test_partial_intersection_one_corner_inside(self):
        dfn_preprocessor = DfnPreprocessor()
        dfn_preprocessor.add_fracture(
            x=0.1, y=0.4, z=0.2, dip=75, dip_dir=65, size=0.7
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
        intersected_points = mesh_preprocessor.elements[0].intersect_with_fracture(
            dfn_preprocessor[0])

        solution = [
            np.array([0.16664204, 0.471432, -0.13807404]),
            np.array([0.25949777, 0.08819276, 0.15230947]),
            np.array([0.33001579, 0.12107585, -0.13807404]),
        ]
        nptest.assert_array_almost_equal(intersected_points, solution)


    def test_no_intersection_fracture_inside(self):
        dfn_preprocessor = DfnPreprocessor()
        dfn_preprocessor.add_fracture(
            x=0.2, y=0.2, z=0.0, dip=60, dip_dir=45, size=0.2
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
        intersected_points = mesh_preprocessor.elements[0].intersect_with_fracture(
            dfn_preprocessor[0])

        solution = [
            np.array([0.30606602, 0.16464466, -0.08660254]),
            np.array([0.16464466, 0.30606602, -0.08660254]),
            np.array([0.09393398, 0.23535534, 0.08660254]),
            np.array([0.23535534, 0.09393398, 0.08660254]),
        ]
        nptest.assert_array_almost_equal(intersected_points, solution)

    def test_intersect_hexahedra_full(self):
        dfn_preprocessor = DfnPreprocessor()
        dfn_preprocessor.add_fracture(
            x=0.0, y=0.0, z=0.0, dip=1, dip_dir=0, size=2.0
        )
        mesh_preprocessor = MeshPreprocessor()
        mesh_preprocessor.add_hexahedra(
            node_ids=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            node_coords=[
                np.array([-0.5, -0.5, -0.5]),
                np.array([0.5, -0.5, -0.5]),
                np.array([0.5, 0.5, -0.5]),
                np.array([-0.5, 0.5, -0.5]),
                np.array([-0.5, -0.5, 0.5]),
                np.array([0.5, -0.5, 0.5]),
                np.array([0.5, 0.5, 0.5]),
                np.array([-0.5, 0.5, 0.5])
            ],
        )

        intersections = mesh_preprocessor.elements[0].intersect_with_fracture(
            dfn_preprocessor[0])
        solution = [
            np.array([-0.5, -0.5, 0.00872753]),
            np.array([-0.5, 0.5, -0.00872753]),
            np.array([0.5, -0.5, 0.00872753]),
            np.array([0.5, 0.5, -0.00872753]),
        ]
        np.testing.assert_array_almost_equal(intersections, solution)





if __name__ == "__main__":
    unittest.main()
