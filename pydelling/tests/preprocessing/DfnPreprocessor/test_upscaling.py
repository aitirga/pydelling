import unittest
import numpy.testing as nptest
import numpy as np
from pydelling.utils.geometry import Line
from pydelling.preprocessing import DfnPreprocessor, MeshPreprocessor
from pydelling.preprocessing.dfn_preprocessor import DfnUpscaler


class UpscalingCase(unittest.TestCase):
    def test_upscale_hexahedra(self):
        dfn_preprocessor = DfnPreprocessor()
        dfn_preprocessor.add_fracture(
            x=0.0, y=0.0, z=0.0, dip=0, dip_dir=0, size=2.0
        )
        dfn_preprocessor.add_fracture(
            x=0.1, y=0.0, z=0.1, dip=0, dip_dir=0, size=0.5
        )
        dfn_preprocessor[0].aperture = 0.01
        dfn_preprocessor[1].aperture = 0.01

        dfn_preprocessor.to_obj('fracture.obj')
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

        dfn_upscaler = DfnUpscaler(
            dfn=dfn_preprocessor,
            mesh=mesh_preprocessor,
            save_intersections=True,
        )

        # Upscale porosity
        porosity = dfn_upscaler.upscale_mesh_porosity()

        # Upscale permeability

        permeability = dfn_upscaler.upscale_mesh_permeability()

        porosity_solution = 0.0125
        #self.assertEqual(porosity, porosity_solution)
        self.assertAlmostEqual(porosity[0], porosity_solution)
        permeability_solution = np.array([[114.70037453,0,0],[0,114.70037453,0],[0,0,0]])
        nptest.assert_array_almost_equal(permeability[0], permeability_solution)



if __name__ == '__main__':
    unittest.main()