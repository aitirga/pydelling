import unittest
import numpy.testing as nptest
import numpy as np
from pydelling.utils.geometry import Line, Plane, Vector


class PlaneCase(unittest.TestCase):
    def test_init(self):
        p = Plane(point=[0, 0, 0], normal=Vector([0, 0, 1]))

    def test_intersection(self):
        plane_1 = Plane(point=[0, 0, 0], normal=Vector([0, 0, 1]))
        plane_2 = Plane(point=[0, 0, 0], normal=Vector([0, 1, 0]))
        intersected_line = plane_1.intersect(plane_2)
        nptest.assert_array_equal(intersected_line.direction_vector, [-1, 0, 0])




if __name__ == '__main__':
    unittest.main()