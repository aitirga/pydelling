from pydelling.preprocessing.GiDPreprocessor import *

if __name__ == '__main__':
    point_1 = Point(coords=[0.0, 0.0, 0.0])
    point_2 = Point(coords=[1.0, 0.0, 0.0])
    point_3 = Point(coords=[0.0, 1.0, 0.0])
    point_4 = Point(coords=[1.0, 1.0, 0.0])

    class LineTest(GidObject):
        def __init__(self, point_1, point_2):
            super().__init__()
            self.point_1 = point_1
            self.point_2 = point_2

        def construct(self):
            self.add([self.point_1, self.point_2])
            test_line = Line(self.point_1, self.point_2)
            self.add(test_line)


    class Combine(GidObject):
        def construct(self, *args, **kwargs):
            line_1 = LineTest(point_1, point_2)
            line_2 = LineTest(point_3, point_4)
            self.add(line_1)
            self.add(line_2)

    combined = Combine()
    combined.run()


