import unittest
import numpy.testing as nptest
import numpy as np
from PyFLOTRAN.utils.geometry import Line, Plane, Vector


class PlaneCase(unittest.TestCase):
    def test_init(self):
        p = Plane(point=[0, 0, 0], normal=Vector([0, 0, 1]))
        print(p)




if __name__ == '__main__':
    unittest.main()