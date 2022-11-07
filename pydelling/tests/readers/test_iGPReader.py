import unittest
from pydelling.readers import iGPReader
from pydelling.config import config
from pydelling.utils import test_data_path
import numpy as np


class iGPReaderCase(unittest.TestCase):
    """
    This class tests the iGPReader[pydelling.readers.iGPReader.io.iGPReader.iGPReader] class implementation
    """
    def setUp(self) -> None:
        self.igp_reader = iGPReader(test_data_path() / "test_implicit_to_explicit.gid",
                                            project_name="test-implicit_to_explicit"
                                            )
        self.igp_reader.build_mesh_data()


    def test_implicit_to_explicit(self):
        self.igp_reader.implicit_to_explicit()

    def test_limits(self):
        min_x = self.igp_reader.min_x
        min_nodes_x = self.igp_reader.coords_min_x
        explicit_min_x = np.min(self.igp_reader.centroids[:, 0])
        explicit_min_x_nodes = np.min(self.igp_reader.nodes[:, 0])

        self.assertEqual(min_x, explicit_min_x)
        self.assertEqual(min_nodes_x, explicit_min_x_nodes)

        max_x = self.igp_reader.max_x
        max_nodes_x = self.igp_reader.coords_max_x
        explicit_max_x = np.max(self.igp_reader.centroids[:, 0])
        explicit_max_x_nodes = np.max(self.igp_reader.nodes[:, 0])

        self.assertEqual(max_x, explicit_max_x)
        self.assertEqual(max_nodes_x, explicit_max_x_nodes)

        min_y = self.igp_reader.min_y
        min_nodes_y = self.igp_reader.coords_min_y
        explicit_min_y = np.min(self.igp_reader.centroids[:, 1])
        explicit_min_y_nodes = np.min(self.igp_reader.nodes[:, 1])

        self.assertEqual(min_y, explicit_min_y)
        self.assertEqual(min_nodes_y, explicit_min_y_nodes)

        max_y = self.igp_reader.max_y
        max_nodes_y = self.igp_reader.coords_max_y
        explicit_max_y = np.max(self.igp_reader.centroids[:, 1])
        explicit_max_y_nodes = np.max(self.igp_reader.nodes[:, 1])

        self.assertEqual(max_y, explicit_max_y)
        self.assertEqual(max_nodes_y, explicit_max_y_nodes)

        min_z = self.igp_reader.min_z
        min_nodes_z = self.igp_reader.coords_min_z
        explicit_min_z = np.min(self.igp_reader.centroids[:, 2])
        explicit_min_z_nodes = np.min(self.igp_reader.nodes[:, 2])

        self.assertEqual(min_z, explicit_min_z)
        self.assertEqual(min_nodes_z, explicit_min_z_nodes)

        max_z = self.igp_reader.max_z
        max_nodes_z = self.igp_reader.coords_max_z
        explicit_max_z = np.max(self.igp_reader.centroids[:, 2])
        explicit_max_z_nodes = np.max(self.igp_reader.nodes[:, 2])

        self.assertEqual(max_z, explicit_max_z)
        self.assertEqual(max_nodes_z, explicit_max_z_nodes)

    def test_properties(self):
        region_names = self.igp_reader.region_names
        self.assertEqual(region_names, ['Pwall'])
        material_names = self.igp_reader.material_names
        self.assertEqual(material_names, [])

        # Get Pwall region centroids
        region_centroids = self.igp_reader.get_region_centroids('Pwall')
        self.assertEqual(region_centroids.shape, (130, 3))
        # Get Pwall region nodes
        region_nodes = self.igp_reader.get_region_nodes('Pwall')
        self.assertEqual(region_nodes.shape, (82, 3))

    def test_boundary_faces(self):
        boundary_faces = self.igp_reader.boundary_names
        self.assertEqual(boundary_faces, ['Pwall'])
        boundary_elements = self.igp_reader.get_boundary_faces('Pwall')
        self.assertEqual(len(boundary_elements), 130)

    def test_region_elements(self):
        self.igp_reader.build_mesh_data()
        top_elements = self.igp_reader.elements
        print(top_elements[0])

if __name__ == '__main__':
    unittest.main()
