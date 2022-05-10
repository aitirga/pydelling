
import numpy.testing as nptest
import numpy as np
from pydelling.utils.geometry import Line, Plane, Vector, Point

if __name__ == '__main__':
    # plane_1 = Plane(point=[0, 0, 3], normal=Vector([0, 0, 1]))
    # plane_2 = Plane(point=[0, -10, 0], normal=Vector([0, -1, 0]))
    # print(plane_1.intersect(plane_2))
    # line_1 = Line(p1=[0, 0, 0], direction_vector=[-1, 0, 0])
    # line_2 = Line(p1=[0.5, 0.5, 0], direction_vector=[1, -1, 0])
    # print(line_1.intersect(line_2))

    # line_1 = Line(p1=[0, 0, 0], direction_vector=[-1, 0, 0])
    # line_2 = Line(p1=[-1, 0, 0], direction_vector=[0, 1, 0])
    # print(line_1.intersect(line_2))
    # test_point: Point = Point([0, 0, 0])
    test_vector: Vector = Vector(p1=[-1, 0, 0], p2=[1, 1, 1])

