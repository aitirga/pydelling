import unittest
from pydelling.preprocessing.gid_preprocessor import *


class GidPreprocessorTest(unittest.TestCase):
    def test_create_line(self):
        point_1 = Point([0.0, 0.0, 0.0])
        point_2 = Point([1.0, 0.0, 0.0])
        line = Line(point_1, point_2)
        class TestGidobject(GidObject):
            def construct(self, *args, **kwargs):
                self.add([point_1, point_2])
                self.add(line)

        test_gid = TestGidobject()
        test_gid.run(internal=True)

    def test_polyline(self):
        point_1 = Point([0.0, 0.0, 0.0])
        point_2 = Point([1.0, 0.0, 0.0])
        point_3 = Point([1.0, 1.0, 0.0])
        point_4 = Point([0.0, 1.0, 0.0])
        polyline = Polyline([point_1, point_2, point_3, point_4], connect=True)
        polyline.run()

    def test_surface(self):
        point_1 = Point([0.0, 0.0, 0.0])
        point_2 = Point([1.0, 0.0, 0.0])
        point_3 = Point([1.0, 1.0, 0.0])
        point_4 = Point([0.0, 1.0, 0.0])

        polyline = Polyline([point_1, point_2, point_3, point_4], connect=True)
        surface = Surface(polyline.lines)

        class TestSurface(GidObject):
            def construct(self, *args, **kwargs):
                self.add(polyline)
                self.add(surface)

        test_surface = TestSurface()
        test_surface.run()

    def test_overlapping_points(self):
        point_1 = Point([0.0, 0.0, 0.0])
        point_2 = Point([0.0, 0.0, 0.0])
        point_3 = Point([1.0, 0.0, 0.0])
        point_4 = Point([2.0, 0.0, 0.0])
        point_5 = Point([1.0, 0.0, 0.0])
        point_6 = Point([2.0, 0.0, 0.0])

        class TestPoints(GidObject):
            def construct(self, *args, **kwargs):
                self.add(point_1)
                self.add(point_2)
                self.add(point_3)
                self.add(point_4)
                self.add(point_5)
                self.add(point_6)
                line_1 = Line(point_1, point_6)
                self.add(line_1)

        test_points = TestPoints()
        test_points.run()

    def test_double_surface(self):
        point_1 = Point([0.0, 0.0, 0.0])
        point_2 = Point([1.0, 0.0, 0.0])
        point_3 = Point([1.0, 1.0, 0.0])
        point_4 = Point([0.0, 1.0, 0.0])
        point_5 = Point([0.0, 2.0, 0.0])

        polyline = Polyline([point_1, point_2, point_3, point_4], connect=True)

        polyline_2 = Polyline([point_1, point_2, point_3, point_5], connect=True)

        surface = Surface(polyline.lines)
        surface_2 = Surface(polyline_2.lines)
        class TestSurface(GidObject):
            def construct(self, *args, **kwargs):
                self.add(polyline)
                self.add(polyline_2)

                self.add(surface)
                self.add(surface_2)

        test_surface = TestSurface()
        test_surface.run()

    def test_lines_equal(self):
        point_1 = Point([0.0, 0.0, 0.0])
        point_2 = Point([1.0, 0.0, 0.0])
        point_3 = Point([1.0, 1.0, 0.0])

        line_1 = Line(point_1, point_2)
        line_2 = Line(point_1, point_2)
        line_3 = Line(point_1, point_3)
        line_4 = Line(point_2, point_1)

        self.assertTrue(Line.check_lines_equal(line_1, line_2))
        self.assertFalse(Line.check_lines_equal(line_1, line_3))
        self.assertTrue(Line.check_lines_equal(line_1, line_4))


if __name__ == '__main__':
    unittest.main()