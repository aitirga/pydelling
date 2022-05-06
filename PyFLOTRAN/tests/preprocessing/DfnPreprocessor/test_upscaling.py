import unittest
import numpy.testing as nptest
import numpy as np
from PyFLOTRAN.utils.geometry import Line
from PyFLOTRAN.preprocessing import DfnPreprocessor, MeshPreprocessor
from PyFLOTRAN.preprocessing.DfnPreprocessor import DfnUpscaler


class UpscalingCase(unittest.TestCase):
    def test_upscale_hexahedra(self):
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

        solution = 1.00
        self.assertEqual(1.0, solution)
        # nptest.assert_array_almost_equal(intersected_points, solution)



if __name__ == '__main__':
    unittest.main()