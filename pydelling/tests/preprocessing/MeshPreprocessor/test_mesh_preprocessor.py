import unittest

import numpy as np

from pydelling.preprocessing.mesh_preprocessor import MeshPreprocessor
from pydelling.readers.FemReader import FemReader
from pydelling.utils import test_data_path
from pydelling.preprocessing import DfnPreprocessor


class TestMeshPreprocessor(unittest.TestCase):
    def test_read_data(self):
        pass

    def test_intersection(self):
        fem_reader = FemReader(test_data_path() / "fem_reader_data.fem")
        dfn_test = DfnPreprocessor()
        # dfn_test.add_fracture(dip=45, dip_dir=0, x=5.0, y=5.0, z=-0.5, size=1)
        dfn_test.add_fracture(dip=45, dip_dir=0, x=5.0, y=5.0, z=-0.5, size=1.5)
        dfn_test.to_obj(filename='./test.obj')
        fem_reader.to_vtk(filename='./test.vtk')
        # fem_reader.find_the_intersection_between_fracture_and_mesh(dfn_test[0])
        fem_reader.find_intersection_points_between_fracture_and_mesh(dfn_test[0])

    def test_read_pyramid(self):
        mesh_preprocessor = MeshPreprocessor()
        mesh_preprocessor.add_pyramid(node_ids=np.array([0, 1, 2, 3, 4]),
                                      node_coords=[np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                                                   np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                                                   np.array([0.0, 0.0, 1.0])])
        self.assertEqual(mesh_preprocessor.n_nodes, 5)


    def test_normal_vector(self):
        pass



if __name__ == '__main__':
    unittest.main()