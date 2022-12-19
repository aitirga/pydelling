import unittest
import numpy.testing as nptest
import numpy as np
from pydelling.utils.geometry import Line
from pydelling.preprocessing import DfnPreprocessor, MeshPreprocessor
from pydelling.preprocessing.dfn_preprocessor import DfnUpscaler
from pydelling.utils import test_data_path

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
        intersected_points = mesh_preprocessor.elements[0].intersect_with_fracture(
            dfn_preprocessor[0])
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
        intersected_points = mesh_preprocessor.elements[0].intersect_with_fracture(
            dfn_preprocessor[0])
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
        intersected_points = mesh_preprocessor.elements[0].intersect_with_fracture(
            dfn_preprocessor[0])

        solution = [
            np.array([0.1,0.,-0.25]),
            np.array([0.1 ,  0.65 ,-0.25]),
            np.array([0.1 ,0. , 0.4]),
        ]
        # dfn_preprocessor.to_vtk('test_dfn.vtk')
        # mesh_preprocessor.to_vtk('test_element.vtk')
        # intersected_points = mesh_preprocessor.elements[0].intersect_with_fracture(
        #     dfn_preprocessor[0], export_all_points=False)
        # print(intersected_points)
        # with open('computed_intersections.csv', 'w') as f:
        #     import csv
        #     writer = csv.writer(f)
        #     writer.writerows(intersected_points)

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
        intersected_points = mesh_preprocessor.elements[0].intersect_with_fracture(
            dfn_preprocessor[0])

        solution = [
            np.array([0.30360464 , 0.      ,   -0.22444437]),
            np.array([0.26325587 , 0.4611885 , -0.22444437]),
            np.array([0.14876171 ,0.   ,      0.35123829]),
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
        intersected_points = mesh_preprocessor.elements[0].intersect_with_fracture(
            dfn_preprocessor[0])

        solution = [
            np.array([0.33225466 , 0.   ,      -0.32785179]),
            np.array([0.   ,       0.71252241, -0.32785179]),
            np.array([0.12422918 ,0.   ,      0.37577082]),
            np.array([0.    ,      0.51272544 ,-0.01272544]),
        ]

        # dfn_preprocessor.to_vtk('test_dfn.vtk')
        # mesh_preprocessor.to_vtk('test_element.vtk')
        # intersected_points = mesh_preprocessor.elements[0].intersect_with_fracture(
        #     dfn_preprocessor[0], export_all_points=False)
        # print(intersected_points)
        # with open('computed_intersections.csv', 'w') as f:
        #     import csv
        #     writer = csv.writer(f)
        #     writer.writerows(intersected_points)

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
        # Sort arrays before comparing
        intersected_points = np.sort(intersected_points, axis=0)
        solution = np.sort(solution, axis=0)
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
        solution = np.array([
            np.array([-0.5, -0.5, 0.00872753]),
            np.array([-0.5, 0.5, -0.00872753]),
            np.array([0.5, -0.5, 0.00872753]),
            np.array([0.5, 0.5, -0.00872753]),
        ])
        # Sort intersections to make them comparable
        intersections = np.sort(intersections, axis=0)
        solution = np.sort(solution, axis=0)

        nptest.assert_array_almost_equal(intersections, solution)

    def test_problematic_fracture_1(self):
        import pickle
        dfn_preprocessor = DfnPreprocessor()
        mesh = MeshPreprocessor()
        read_fracture = test_data_path() / "problematic_fractures/issue_fracture_1.pkl"
        with open(read_fracture, "rb") as f:
            fracture = pickle.load(f)
        read_element = test_data_path() / "problematic_fractures/issue_element_1.pkl"
        with open(read_element, "rb") as f:
            element = pickle.load(f)

        dfn_preprocessor.dfn.append(fracture)
        dfn_preprocessor.to_vtk('test_dfn.vtk')
        element.to_obj('test_element.obj')
        intersections = element.intersect_with_fracture(dfn_preprocessor[0], export_all_points=False)
        self.assertEqual(len(intersections), 3)

    def test_problematic_fracture_2(self):
        import pickle
        dfn_preprocessor = DfnPreprocessor()
        mesh = MeshPreprocessor()
        read_fracture = test_data_path() / "problematic_fractures/issue_fracture_2.pkl"
        with open(read_fracture, "rb") as f:
            fracture = pickle.load(f)
        read_element = test_data_path() / "problematic_fractures/issue_element_2.pkl"
        with open(read_element, "rb") as f:
            element = pickle.load(f)

        dfn_preprocessor.dfn.append(fracture)
        mesh.elements.append(element)
        intersections = mesh.elements[0].intersect_with_fracture(dfn_preprocessor[0])
        self.assertEqual(len(intersections), 4)

    def test_prism_intersection(self):
        dfn_preprocessor = DfnPreprocessor()
        dfn_preprocessor.add_fracture(
            x=0.0, y=0.0, z=0.0, dip=1, dip_dir=0, size=2.0
        )
        mesh_preprocessor = MeshPreprocessor()
        mesh_preprocessor.add_wedge(
            node_ids=np.array([0, 1, 2, 3, 4, 5]),
            node_coords=[
                np.array([-0.5, -0.5, -0.5]),
                np.array([0.5, -0.5, -0.5]),
                np.array([0.5, 0.5, -0.5]),
                np.array([-0.5, -0.5, 0.5]),
                np.array([0.5, -0.5, 0.5]),
                np.array([0.5, 0.5, 0.5]),
            ],
        )

        intersections = mesh_preprocessor.elements[0].intersect_with_fracture(
            dfn_preprocessor[0])
        solution = np.array([
            np.array([-0.5, -0.5, 0.00872753]),
            np.array([0.5, -0.5, 0.00872753]),
            np.array([0.5, 0.5, -0.00872753]),
        ])
        # Sort intersections to make them comparable
        intersections = np.sort(intersections, axis=0)
        solution = np.sort(solution, axis=0)

        nptest.assert_array_almost_equal(intersections, solution)


    def tearDown(self) -> None:
        from pathlib import Path
        for file in Path().glob('*.pkl'):
            file.unlink()
        for file in Path().glob('*.obj'):
            file.unlink()
        for file in Path().glob('*.vtk'):
            file.unlink()
        for file in Path().glob('*.json'):
            file.unlink()







if __name__ == "__main__":
    unittest.main()
