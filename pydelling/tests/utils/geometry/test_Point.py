import unittest
import numpy.testing as nptest
import numpy as np
from pydelling.utils.geometry import Point


class PointCase(unittest.TestCase):
    def test_Point_init(self):
        p = Point([1,2,3])
        self.assertEqual(p.x, 1)
        self.assertEqual(p.y, 2)
        self.assertEqual(p.z, 3)

    def test_distance_point(self):
        p1: Point = Point([0, 0, 0])
        p2: Point = Point([1, 1, 0])
        self.assertEqual(p1.distance(p2), np.sqrt(2))


if __name__ == '__main__':
    unittest.main()
