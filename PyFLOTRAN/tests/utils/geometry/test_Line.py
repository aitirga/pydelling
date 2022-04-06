import unittest
import numpy.testing as nptest
import numpy as np
from PyFLOTRAN.utils.geometry import Line


class LineCase(unittest.TestCase):
    def test_line_init(self):
        line = Line(np.array([0,0,0]), np.array([1,1,0]))
        nptest.assert_array_almost_equal(line.p.coords, np.array([0, 0, 0]))
        nptest.assert_array_almost_equal(line.direction_vector, np.array([np.sqrt(2) / 2.0,np.sqrt(2) / 2.0, 0]))
        nptest.assert_array_equal(line.p, np.array([0, 0, 0]))

    def test_parallel(self):
        line_1 = Line(np.array([0,0,0]), np.array([1,1,0]))
        line_2 = Line(np.array([0,0,0]), np.array([2,2,0]))
        self.assertTrue(line_1.is_parallel(line_2))

    def test_not_parallel(self):
        line_1 = Line(np.array([0,0,0]), np.array([1,1,0]))
        line_2 = Line(np.array([0,0,0]), np.array([3,2,0]))
        self.assertFalse(line_1.is_parallel(line_2))

    def test_intersection_1(self):
        line_1 = Line(np.array([0,0,0]), np.array([-1,0,0]))
        line_2 = Line(np.array([-1,2,0]), np.array([-1,1,0]))
        intersection_point = line_1.intersect(line_2)
        nptest.assert_array_almost_equal(intersection_point, np.array([-1,0,0]))

    def test_intersection_2(self):
        line_1 = Line(np.array([0,0,0]), np.array([2,2,0]))
        line_2 = Line(np.array([2,0,0]), direction_vector=np.array([-1,1,0]))
        intersection_point = line_1.intersect(line_2)
        nptest.assert_array_almost_equal(intersection_point, np.array([1,1,0]))

    def test_intersection_3(self):
        line_1 = Line(np.array([0,0,0]), np.array([0,1,0]))
        line_2 = Line(np.array([0,-1,2]), np.array([0,-1,1]))
        intersection_point = line_1.intersect(line_2)
        nptest.assert_array_almost_equal(intersection_point, np.array([0,-1,0]))


if __name__ == '__main__':
    unittest.main()