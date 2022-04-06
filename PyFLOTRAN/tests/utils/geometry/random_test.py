import numpy.testing as nptest
import numpy as np
from PyFLOTRAN.utils.geometry import Line, Plane, Vector

if __name__ == '__main__':
    plane_1 = Plane(point=[0, 0, 0], normal=Vector([0, 0, 1]))
    plane_2 = Plane(point=[0, 0, 0], normal=Vector([0, 1, 0]))
    print(plane_1.intersect(plane_2))