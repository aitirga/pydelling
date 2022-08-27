from typing import List

import numpy as np
import shapely.geometry as geom

from pydelling.utils.geometry import Plane, Segment, Point, Line


class PolygonFracture:
    local_id = 0
    eps = 1e-8
    def __init__(self, dip, dip_dir, x, y, z, size, aperture=0.01):
        # super().__init__(dip, dip_dir, x, y, z, size, aperture)
        self.side_points = None
        self.dip = dip
        self.dip_dir = dip_dir
        self.x_centroid = x
        self.y_centroid = y
        self.z_centroid = z
        self.size = size
        self.intersection_dictionary = {}
        self.aperture = aperture
        self.local_id = PolygonFracture.local_id
        PolygonFracture.local_id += 1

    def get_side_points_v1(self):
        """
        Finds the side points of the fracture
        Returns: Coordinates of the plane side points
        """
        phi = self.dip_dir / 360 * (2 * np.pi)
        theta = self.dip / 360 * (2 * np.pi)

        u = np.array([np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi), -np.sin(theta)])
        v = np.array([np.cos(phi), - np.sin(phi), 0])

        A = self.centroid + self.size / 2 * (u + v)
        B = self.centroid + self.size / 2 * (u - v)
        C = self.centroid - self.size / 2 * (u + v)
        D = self.centroid - self.size / 2 * (u - v)


        self.side_points = 4

        return np.array([A, B, C, D])


    def get_side_points_v3(self):
        """
        alpha = 45; %strike
        delta = 30; %dip
        w = 4; %width (dipping side)
        L = 9; %Length (side parallel to the surface)

        A = [0,0,0]; % Pivot point at the surface

        H = w*sind(alpha+90);
        V = w*cosd(alpha+90);
        Z = -w*sind(delta);

        B = [L*sind(alpha) L*cosd(alpha) 0]+A;
        C = [L*cosd(90-alpha)+H L*sind(90-alpha)+V Z]+A;
        D = [w*sind(alpha+90) w*cosd(alpha+90) -w*sind(delta)]+A;

        P = [A; B; C; D];
        """
        alpha = self.dip_dir / 360 * (2 * np.pi)
        delta = self.dip / 360 * (2 * np.pi)
        w = self.size
        L = self.size

        A = np.array(self.centroid)
        A[0] = A[0] - np.sin(alpha) * L / 2
        A[1] = A[1] + np.cos(alpha) * L / 2
        A[2] = A[2] + np.sin(delta) * L / 2


        H = w * np.sin(alpha + np.pi / 2)
        V = w * np.cos(alpha + np.pi / 2)
        Z = -w * np.sin(delta)

        B = np.array([L * np.sin(alpha), L * np.cos(alpha), 0]) + A
        C = np.array([L * np.cos(np.pi / 2 - alpha) + H, L * np.sin(np.pi / 2 - alpha) + V, Z]) + A
        D = np.array([w * np.sin(alpha + np.pi / 2), w * np.cos(alpha + np.pi / 2), -w * np.sin(delta)]) + A

        P = np.array([A, B, C, D])

        return P

    def get_side_points_v2(self):
        """
        Finds the side points of the fracture
        Returns: Coordinates of the plane side points
        """
        alpha = self.dip / 360 * (2 * np.pi)
        beta = self.dip_dir / 360 * (2 * np.pi)

        A = self.centroid + np.array([
            + self.size / 2 * (-np.cos(beta) - np.sin(beta) * np.cos(alpha)),
            + self.size / 2 * (np.sin(beta) - np.cos(beta) * np.cos(alpha)),
            self.size / 2 * np.sin(alpha)
        ])

        B = self.centroid + np.array([
            + self.size / 2 * (-np.cos(beta) + np.sin(beta) * np.cos(alpha)),
            + self.size / 2 * (np.sin(beta) + np.cos(beta) * np.cos(alpha)),
            - self.size / 2 * np.sin(alpha)
        ])

        C = self.centroid + np.array([
            + self.size / 2 * (np.cos(beta) + np.sin(beta) * np.cos(alpha)),
            + self.size / 2 * (-np.sin(beta) + np.cos(beta) * np.cos(alpha)),
            - self.size / 2 * np.sin(alpha)
        ])

        D = self.centroid + np.array([
            + self.size / 2 * (np.cos(beta) - np.sin(beta) * np.cos(alpha)),
            + self.size / 2 * (-np.sin(beta) - np.cos(beta) * np.cos(alpha)),
            + self.size / 2 * np.sin(alpha)
        ])

        self.side_points = 4

        return np.array([A, B, C, D])


    def get_side_points(self, method='v1'):
        if method == 'v1':
            return self.get_side_points_v1()
        elif method == 'v2':
            return self.get_side_points_v2()
        elif method == 'v3':
            return self.get_side_points_v3()


    def to_obj(self, global_id=0, method='v1'):
        """Converts the fracture to an obj file"""
        side_points = self.get_side_points(method=method)
        obj_string = ''
        for i in range(len(side_points)):
            obj_string += 'v ' + str(side_points[i][0]) + ' ' + str(side_points[i][1]) + ' ' + str(side_points[i][2]) + '\n'
        obj_string += f'f '
        for i in range(len(side_points)):
            obj_string += str(global_id + i) + ' '
        obj_string += '\n'
        return obj_string

    @property
    def unit_normal_vector(self):
        """Returns the normal vector of the fracture"""
        get_side_points = self.get_side_points()
        v1 = get_side_points[1] - get_side_points[0]
        v2 = get_side_points[2] - get_side_points[0]
        cross = np.cross(v1, v2)
        return cross / np.linalg.norm(cross)

    def distance_to_point(self, point: np.ndarray):
        """Returns the distance to a point"""
        distance_vector = self.centroid - point
        return np.dot(distance_vector, self.unit_normal_vector)

    def get_bounding_box(self):
        """Returns the bounding box of the fracture"""
        side_points = self.get_side_points()
        x_min = np.min(side_points[:, 0])
        x_max = np.max(side_points[:, 0])
        y_min = np.min(side_points[:, 1])
        y_max = np.max(side_points[:, 1])
        z_min = np.min(side_points[:, 2])
        z_max = np.max(side_points[:, 2])
        return np.array([x_min, x_max, y_min, y_max, z_min, z_max])

    def point_inside_bounding_box(self, point: np.ndarray):
        """Returns if a point is inside the bounding box of the fracture"""
        bounding_box = self.get_bounding_box()
        if point[0] < bounding_box[0] or point[0] > bounding_box[1]:
            return False
        elif point[1] < bounding_box[2] or point[1] > bounding_box[3]:
            return False
        elif point[2] < bounding_box[4] or point[2] > bounding_box[5]:
            return False
        else:
            return True

    @property
    def centroid(self):
        return np.array([self.x_centroid, self.y_centroid, self.z_centroid])

    @property
    def polygon(self):
        side_points = self.get_side_points()
        self._polygon = geom.Polygon(side_points)
        return self._polygon


    @property
    def plane(self):
        return Plane(self.centroid, normal=self.unit_normal_vector)

    @property
    def corners(self) -> List[Point]:
        """Returns the corners of the fracture"""
        return [Point(point) for point in self.get_side_points()]

    @property
    def corner_segments(self):
        """Returns the corner lines of the fracture"""
        corner_segments = [
            Segment(self.corners[0], self.corners[1]),
            Segment(self.corners[1], self.corners[2]),
            Segment(self.corners[2], self.corners[3]),
            Segment(self.corners[3], self.corners[0])
        ]
        return corner_segments

    @property
    def corner_lines(self):
        """Returns the corner lines of the fracture"""
        corner_segments = [
            Line(self.corners[0], self.corners[1]),
            Line(self.corners[1], self.corners[2]),
            Line(self.corners[2], self.corners[3]),
            Line(self.corners[3], self.corners[0])
        ]
        return corner_segments

    def contains(self, point: Point):
        """Returns if a point is inside the fracture"""
        q1, q2, q3, q4 = self.corners
        q1: Point

        largest_normal_index = self.largest_index_normal_vector
        q1_hat = np.delete(q1, largest_normal_index)
        q2_hat = np.delete(q2, largest_normal_index)
        q3_hat = np.delete(q3, largest_normal_index)
        q4_hat = np.delete(q4, largest_normal_index)
        p_hat = np.delete(point, largest_normal_index)
        u0 = p_hat[0]
        u1 = q1_hat[0]
        u2 = q2_hat[0]
        u3 = q3_hat[0]
        u4 = q4_hat[0]
        v0 = p_hat[1]
        v1 = q1_hat[1]
        v2 = q2_hat[1]
        v3 = q3_hat[1]
        v4 = q4_hat[1]


        s1 = (v1 - v2) * u0 + (u2 - u1) * v0 + v2 * u1 - u2 * v1
        s2 = (v2 - v3) * u0 + (u3 - u2) * v0 + v3 * u2 - u3 * v2
        s3 = (v3 - v4) * u0 + (u4 - u3) * v0 + v4 * u3 - u4 * v3
        s4 = (v4 - v1) * u0 + (u1 - u4) * v0 + v1 * u4 - u1 * v4

        s = np.array([s1, s2, s3, s4])

        equal_sign = np.all(s >= -self.eps) if s[0] >= -self.eps else np.all(s <= self.eps)

        if equal_sign:
            return True
        else:
            return False

    @property
    def largest_index_normal_vector(self):
        """Returns the largest coordinate index of the normal vector"""
        return np.argmax(self.unit_normal_vector)






