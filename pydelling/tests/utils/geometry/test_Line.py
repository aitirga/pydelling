import unittest
import numpy.testing as nptest
import numpy as np
from pydelling.utils.geometry import Line, Plane


class LineCase(unittest.TestCase):
    def test_line_init(self):
        line = Line(np.array([0,0,0]), np.array([1,1,0]))
        nptest.assert_array_almost_equal(line.p, np.array([0, 0, 0]))
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

    def test_intersection_4(self):
        line_1 = Line([1, 0, 0], direction_vector=[-1, 0, 1])
        line_2 = Line(p1=[1, 1, 0], p2=[-1, -1, 2])
        line_1_line_2_intersection = line_1.intersect(line_2)
        line_2_line_1_intersection = line_2.intersect(line_1)
        nptest.assert_array_almost_equal(line_2_line_1_intersection, np.array([0, 0, 1]))
        nptest.assert_array_almost_equal(line_1_line_2_intersection, line_2_line_1_intersection)

    def test_intersection_5(self):
        import time
        line_1 = Line([1, 0, 0], direction_vector=[-1, 0, 1])
        line_2 = Line(p1=[1, 1, 0], p2=[-1, -1, 2])

        time_1 = time.time()
        for i in range(1000):
            line_1_line_2_intersection = line_1.intersect(line_2)
        time_2 = time.time()
        delta_time = time_2 - time_1
        self.assertLess(delta_time, 0.5)

    def test_intersect_plane_line(self):
        line = Line(np.array([0,0,0]), np.array([3,0,0]))
        plane = Plane(point=[2.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0])
        intersection_point = line.intersect(plane)
        nptest.assert_array_almost_equal(intersection_point, np.array([2.0, 0.0, 0.0]))


if __name__ == '__main__':
    unittest.main()