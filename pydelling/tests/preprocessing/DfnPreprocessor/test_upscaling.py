import unittest
import numpy.testing as nptest
import numpy as np
from pydelling.utils.geometry import Line
from pydelling.preprocessing import DfnPreprocessor, MeshPreprocessor
from pydelling.preprocessing.dfn_preprocessor import DfnUpscaler


class UpscalingCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dfn_preprocessor = DfnPreprocessor()
        self.dfn_preprocessor.add_fracture(
            x=0.0, y=0.0, z=0.0, dip=0, dip_dir=0, size=2.0
        )
        self.dfn_preprocessor.add_fracture(
            x=0.1, y=0.0, z=0.1, dip=0, dip_dir=0, size=0.5
        )
        self.dfn_preprocessor[0].aperture = 0.01
        self.dfn_preprocessor[1].aperture = 0.01

        self.dfn_preprocessor[0].hydraulic_aperture = 0.01
        self.dfn_preprocessor[1].hydraulic_aperture = 0.01

        self.dfn_preprocessor.to_obj('fracture.obj')
        self.mesh_preprocessor = MeshPreprocessor()
        self.mesh_preprocessor.add_hexahedra(
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
        self.mesh_preprocessor.to_vtk(filename='./test_mesh.vtk')

        self.dfn_upscaler = DfnUpscaler(
            dfn=self.dfn_preprocessor,
            mesh=self.mesh_preprocessor,
        )

    def test_porosity_and_permeability(self):
        porosity = self.dfn_upscaler.upscale_mesh_porosity(intensity_correction_factor=1.53,
                                                           existing_fractures_fraction=0.385,
                                                           truncate_to_min_percentile=0,
                                                           truncate_to_max_percentile=100)

        porosity_solution = 0.049675324675324685
        self.assertAlmostEqual(porosity[0], porosity_solution)
        # Upscale permeability
        permeability = self.dfn_upscaler.upscale_mesh_permeability(truncate_to_min_percentile=0,
                                                                   truncate_to_max_percentile=100)
        # #self.assertEqual(porosity, porosity_solution)
        permeability_solution = np.array([[1.1470037453,0,0],[0,1.1470037453,0],[0,0,0]])
        nptest.assert_array_almost_equal(permeability[0], permeability_solution)

    # def test_save_load_upscaler(self):
    #     self.dfn_upscaler.save('upscaler.pkl')
    #     new_upscaler = DfnUpscaler.load('upscaler.pkl')
    #     self.assertEqual(len(self.dfn_upscaler.mesh.elements), len(new_upscaler.mesh.elements))
    #     self.assertEqual(len(self.dfn_upscaler.dfn.dfn), len(new_upscaler.dfn.dfn))
    #     self.assertEqual(len(self.dfn_upscaler.dfn.dfn[0].intersection_dictionary), len(new_upscaler.dfn.dfn[0].intersection_dictionary))
    #     self.assertEqual(len(self.dfn_upscaler.dfn.faults), len(new_upscaler.dfn.faults))
    #     self.assertEqual(len(self.dfn_upscaler.upscaled_porosity), len(new_upscaler.upscaled_porosity))
    #     from pathlib import Path
    #     Path('upscaler.pkl').unlink()

    # def test_save_load_json(self):
    #     self.dfn_upscaler.to_json('upscaler.json')
    #     new_upscaler = DfnUpscaler.from_json('upscaler.json')
    #     new_upscaler.upscale_mesh_porosity()
    #     # self.dfn_upscaler.upscale_mesh_porosity()
    #     # self.assertEqual(self.dfn_upscaler.upscaled_porosity[0], new_upscaler.upscaled_porosity[0])
    # #


if __name__ == '__main__':
    unittest.main()