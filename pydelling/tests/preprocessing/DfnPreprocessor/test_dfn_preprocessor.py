import unittest
from pydelling.preprocessing import DfnPreprocessor


class DfnPreprocessorCase(unittest.TestCase):
    def test_dfn_sum(self):
        dfn_one = DfnPreprocessor()
        dfn_one.add_fracture(
            x=0, y=0, z=0, dip=0, dip_dir=0, aperture=0.1
        )
        dfn_two = DfnPreprocessor()
        dfn_two.add_fracture(
            x=0, y=0, z=0, dip=0, dip_dir=0, aperture=0.5
        )
        dfn_three = dfn_one + dfn_two
        self.assertEqual(len(dfn_three.dfn), 2)


if __name__ == '__main__':
    unittest.main()