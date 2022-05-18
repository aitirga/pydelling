import unittest
from pydelling.utils.geometry_utils import *
from pydelling.utils.geometry import Point


class GeometryUtilsCase(unittest.TestCase):
    def test_compute_area(self):
        # Test 1
        points = [Point([0.0, 0.0, 0.0]),
                  Point([2.0, 0.0, 0.0]),
                  Point([2.0, 1.0, 0.0]),
                  Point([0.0, 1.0, 0.0])]
        self.assertEqual(compute_polygon_area(points), 2.0)

        # Test 2
        points = [Point([0.0, 0.0, 0.0]),
                  Point([0.0, 2.0, 0.0]),
                  Point([0.0, 2.0, 2.0]),
                  Point([0.0, 0.0, 2.0])]
        self.assertEqual(compute_polygon_area(points), 4.0)

        # Test 3: triangle
        points = [Point([0.0, 0.0, 0.0]),
                  Point([0.0, 2.0, 0.0]),
                  Point([2.0, 0.0, 0.0])
                  ]
        self.assertEqual(compute_polygon_area(points), 2.0)
        # Test 4: real case
        points = [Point([2.5, 5.0, -0.5]),
                  Point([2.89216, 5.0, -1.0]),
                  Point([2.89216, 5.0, -0.820348]),
                  ]

    def test_compute_area_unordered(self):
        # Test 1
        points = [Point([0.0, 1.0, 0.0]),
                  Point([0.0, 1.0, 2.0]),
                  Point([0.0, 0.0, 0.0]),
                  Point([0.0, 0.0, 2.0])]
        self.assertEqual(compute_polygon_area(points), 2.0)

        # Test 2
        points = [Point([0.0, 1.0, 0.0]),
                  Point([2.0, 1.0, 0.0]),
                  Point([2.0, 1.0, 2.0]),
                  Point([0.0, 1.0, 2.0])]
        self.assertEqual(compute_polygon_area(points), 4.0)


if __name__ == '__main__':
    unittest.main()
