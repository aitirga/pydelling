import numpy as np

from . import Line, Point, Plane, Segment


def intersect_line_line(line_1: Line, line_2: Line):
    if line_1.is_parallel(line_2):
        return None
    else:
        delta_p = line_1.p - line_2.p
        a = np.array([
            [line_1.direction_vector[0], -line_2.direction_vector[0]],
            [line_1.direction_vector[1], -line_2.direction_vector[1]],
            [line_1.direction_vector[2], -line_2.direction_vector[2]]

        ])
        b = - np.array([
            [delta_p[0]],
            [delta_p[1]],
            [delta_p[2]],
        ])
        x = np.linalg.lstsq(a, b, rcond=-1)
        if x[1] >= line_1.eps:
            return None
        else:
            return Point(line_1.p + line_1.direction_vector * x[0][0])

def intersect_plane_plane(plane_1: Plane, plane_2: Plane):
    """Performs the intersection of this plane with the given plane"""
    if plane_1.is_parallel(plane_2):
        return None
    normal_a = plane_1.n
    normal_b = plane_2.n
    d_a = np.dot(plane_1.p, plane_1.n)
    d_b = np.dot(plane_2.p, plane_2.n)
    U = np.cross(normal_a, normal_b)
    M = np.array((normal_a, normal_b, U))
    X = np.array([d_a, d_b, 0.0]).reshape(3, 1)
    # print(M)
    p_inter = np.linalg.solve(M, X).T
    p1 = p_inter[0]
    p2 = (p_inter + U)[0]
    intersected_line = Line(p1, p2)

    return intersected_line

def intersect_plane_line(plane: Plane, line: Line):
    """Performs the intersection of this plane with the given line"""
    # The point r = q + lambda * v
    dot_n_diff = np.dot(plane.n, plane.p - line.p)
    dot_n_v = np.dot(plane.n, line.direction_vector)
    if dot_n_v == 0: # Plane and line are parallel
        return None
    else:
        lambda_ = dot_n_diff / dot_n_v
        return Point(line.p + lambda_ * line.direction_vector)

def intersect_plane_segment(plane: Plane, segment: Segment):
    """Performs the intersection of this plane with the given segment"""
    # The point r = q + lambda * v
    dot_n_diff = np.dot(plane.n, plane.p - segment.p1)
    dot_n_v = np.dot(plane.n, segment.direction_vector)
    if dot_n_v == 0: # Plane and line are parallel
        return None
    else:
        lambda_ = dot_n_diff / dot_n_v
        intersection_point = Point(segment.p1 + lambda_ * segment.direction_vector)
        if segment.contains(intersection_point):
            return intersection_point
        else:
            return None





