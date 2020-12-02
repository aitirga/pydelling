import numpy as np


# Useful geometrical functions
def normal_vector(points):
    assert len(points) >= 3, "Incorrect number of points, more are needed to form a plane"
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    vn = np.cross(v1, v2)
    vn_norm = np.linalg.norm(vn)
    return vn / vn_norm


def line_plane_intersection(line_points=[], plane_points=[]):
    # Compute plane normal
    # line points
    p0 = line_points[0]
    p1 = line_points[1]
    u = p1 - p0
    # plane points
    v0 = plane_points[0]
    vn = normal_vector(plane_points)
    s = (np.dot(vn, (v0 - p0)) / np.dot(vn, u))
    intersection_point = p0 + s * u
    return intersection_point


def line_plane_perpendicular_intersection(line_point=None, plane_points=[]):
    # Compute plane normal
    # line points
    p0 = line_point
    # plane points
    v0 = plane_points[0]
    vn = normal_vector(plane_points)
    s = (np.dot(vn, (v0 - p0)) / np.dot(vn, vn))
    intersection_point = p0 + s * vn
    return intersection_point
