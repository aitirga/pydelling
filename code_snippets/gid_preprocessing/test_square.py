from PyFLOTRAN.preprocessing.GiDPreprocessor import *


if __name__ == '__main__':
    class Square(GidObject):
        def set_up(self, point_1, point_2, point_3, point_4):
            self.point_1 = point_1
            self.point_2 = point_2
            self.point_3 = point_3
            self.point_4 = point_4

        def construct(self, *args, **kwargs):
            line_1 = Line(self.point_1, self.point_2)
            line_2 = Line(self.point_2, self.point_3)
            line_3 = Line(self.point_3, self.point_4)
            line_4 = Line(self.point_4, self.point_1)

            self.add([self.point_1, self.point_2, self.point_3, self.point_4])
            self.add([line_1, line_2, line_3, line_4])
            surface = Surface([line_1, line_2, line_3, line_4])
            self.add(surface)
            self.extrude(surface, start_point=Point([0.0, 0.0, 0.0]), end_point=Point([0.0, 0.0, 5.0]))

            point_5 = Point([2.0, 0.0, 0.0])
            point_6 = Point([3.0, 0.0, 0.0])
            line_new = Line(point_5, point_6)
            self.add([point_5, point_6])
            self.add(line_new)


    point_1 = Point([0.0, 0.0, 0.0])
    point_2 = Point([1.0, 0.0, 0.0])
    point_3 = Point([1.0, 1.0, 0.0])
    point_4 = Point([0.0, 1.0, 0.0])

    square = Square(point_1=point_1, point_2=point_2, point_3=point_3, point_4=point_4)
    square.run()





