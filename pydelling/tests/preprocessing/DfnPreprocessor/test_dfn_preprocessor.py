import unittest
from pydelling.preprocessing import DfnPreprocessor
from pydelling.utils import test_data_path
from pydelling.config import config


class DfnPreprocessorCase(unittest.TestCase):
    def test_dfn_sum(self):
        dfn_one = DfnPreprocessor()
        dfn_one.add_fracture(
            x=0, y=0, z=0, dip=0, dip_dir=0, aperture=0.1, size=1.2
        )
        dfn_two = DfnPreprocessor()
        dfn_two.add_fracture(
            x=0, y=0, z=0, dip=0, dip_dir=0, aperture=0.5, size=1.2
        )
        dfn_three = dfn_one + dfn_two
        self.assertEqual(len(dfn_three.dfn), 2)

    def test_dfn_save_load(self):
        dfn_one = DfnPreprocessor()
        dfn_one.add_fracture(
            x=0, y=0, z=0, dip=0, dip_dir=0, aperture=0.1, size=1.2
        )
        dfn_one.to_json('test.json')
        dfn_one_loaded = DfnPreprocessor.from_json('test.json')
        self.assertEqual(len(dfn_one_loaded.dfn), 1)
        self.assertEqual(dfn_one_loaded.dfn[0].x_centroid, 0)
        self.assertEqual(dfn_one_loaded.dfn[0].aperture, 0.1)
        self.assertEqual(dfn_one_loaded.dfn[0].size, 1.2)

    def test_dfn_save_load_with_storativity(self):
        dfn_one = DfnPreprocessor()
        dfn_one.add_fracture(
            x=0, y=0, z=0, dip=0, dip_dir=0, aperture=0.1, size=1.2, rock_type=0, transmissivity_constant={0: 1E-6, 1: 1E-6},
        )
        dfn_one.to_json('test.json')
        dfn_one_loaded = DfnPreprocessor.from_json('test.json')
        self.assertEqual(len(dfn_one_loaded.dfn), 1)
        self.assertEqual(dfn_one_loaded.dfn[0].x_centroid, 0)
        self.assertEqual(dfn_one_loaded.dfn[0].aperture, 0.1)
        self.assertEqual(dfn_one_loaded.dfn[0].size, 1.2)

    def test_fault_save_load(self):
        dfn_one = DfnPreprocessor()
        dfn_one.add_fault(filename=test_data_path() / 'test_fault.stl', aperture=5.0)
        dfn_one.to_json('./test.json')
        dfn_one_loaded = DfnPreprocessor.from_json('test.json')
        self.assertEqual(len(dfn_one_loaded.faults), 1)
        self.assertEqual(dfn_one_loaded.faults[0].aperture, 5.0)

    def tearDown(self) -> None:
        import os
        try:
            os.remove('test.json')
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    unittest.main()