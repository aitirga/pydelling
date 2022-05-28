import numpy as np
from scipy.spatial.qhull import ConvexHull
from shapely.geometry import Polygon

# Useful geometrical functions
def normal_vector(points):
    """Computes normal vector of a co-planar set of points (vertices of a polygon)"""
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


def compute_polygon_area(points) -> float:
    """
    This function computes the area of a 3D planar polygon
    :return: Area of a 3D planar polygon
    """

    if len(points) < 3:
        return 0.0

    # convex_hull = Polygon(points)
    # return convex_hull.convex_hull.area

    # Compute normal
    points = order_points_clockwise(points)
    vn = normal_vector(points)
    temp_area_v = np.zeros(shape=3)
    projected_area = 0.0
    n_coords = len(points)
    for id, point in enumerate(points):
        # Set-up variables
        v1 = points[id % n_coords]  # Pv1, assuming P=(0,0,0)
        v2 = points[(id + 1) % n_coords]  # Pv2, assuming P=(0,0,0)
        # Compute area
        id_area_v = np.cross(v1, v2)
        projected_area_id = np.dot(vn, id_area_v) / 2.0  # area of small triangle of the face
        projected_area += projected_area_id
    return projected_area


def order_points_clockwise(points: np.ndarray) -> list:
    """Orders set of 3D coplanar points clockwise"""
    # Compute normal vector
    vn = normal_vector(points)
    # Compute centroid
    centroid = np.mean(points, axis=0)
    # Project points on normal plane
    v1 = points[0] - centroid
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(vn, v1)
    v2 = v2 / np.linalg.norm(v2)
    projected_points = np.zeros(shape=(len(points), 3))
    for id, point in enumerate(points):
        projected_points[id] = [np.dot(v1, point), np.dot(v2, point), 0.0]
    # Order points clockwise, compute angles
    center = np.mean(projected_points, axis=0)
    angles = np.zeros(shape=len(projected_points))
    for id, point in enumerate(projected_points):
        angles[id] = np.arctan2(point[1] - center[1], point[0] - center[0])
    # Sort points
    sorted_points = [points[id] for id in np.argsort(angles)]
    return sorted_points


