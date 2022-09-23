from typing import List

import numpy as np
import shapely.geometry as geom

from pydelling.utils.geometry import Plane, Segment, Point, Line


class Fracture(object):
    local_id = 0
    eps = 1e-3
    _transmissivity = None
    _storativity = None
    _side_points = None
    _unit_normal_vector = None

    def __init__(self,
                 x=None,
                 y=None,
                 z=None,
                 dip=None,
                 dip_dir=None,
                 size=None,
                 aperture=None,
                 hydraulic_aperture=None,
                 rock_type=None,
                 aperture_constant=None,
                 transmissivity_constant=None,
                 storativity_constant=None,
                 normal_vector=None,
                 polygon: List[np.ndarray]=None,
                 ):
        if normal_vector is not None:
            self._unit_normal_vector: np.ndarray = normal_vector
        self.side_points = None
        if dip is not None:
            assert dip_dir is not None
        if size is None:
            assert aperture is not None
        if aperture is None:
            assert size is not None
        # Allow definition of a Fracture object based on a polygon
        if polygon is not None:
            # Assert x, y, z are not provided
            self._side_points = polygon
            centroid = np.mean(polygon, axis=0)
            x = centroid[0]
            y = centroid[1]
            z = centroid[2]
        self.polygon_points = polygon


        self.dip = dip
        self.dip_dir = dip_dir
        self.x_centroid = x
        self.y_centroid = y
        self.z_centroid = z
        self.size = size
        self._aperture = aperture
        self.hydraulic_aperture = hydraulic_aperture
        self.rock_type = rock_type
        self.intersection_dictionary = {}
        self.aperture_constant = aperture_constant
        if transmissivity_constant is not None:
            if rock_type is not None:
                if isinstance(transmissivity_constant, dict):
                    self.transmissivity_constant = transmissivity_constant[int(rock_type)]
                else:
                    self.transmissivity_constant = transmissivity_constant
            else:
                self.transmissivity_constant = transmissivity_constant
        else:
            self.transmissivity_constant = None
        if storativity_constant is not None:
            if rock_type is not None:
                if isinstance(storativity_constant, dict):
                    self.storativity_constant = storativity_constant[int(rock_type)]
                else:
                    self.storativity_constant = storativity_constant
            else:
                self.storativity_constant = storativity_constant
        else:
            self.storativity_constant = None

        self.aperture = self.compute_aperture()
        self.local_id = Fracture.local_id
        self.side_points = self.get_side_points()
        self.n_side_points = len(self.side_points)


        Fracture.local_id += 1

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
        if self._side_points is not None:
            return self._side_points
        if method == 'v1':
            return self.get_side_points_v1()
        elif method == 'v2':
            return self.get_side_points_v2()
        elif method == 'v3':
            return self.get_side_points_v3()


    def to_obj(self, global_id=0, method='v1'):
        """Converts the fracture to an obj file"""
        side_points = self.get_side_points(method=method)
        if isinstance(side_points[0], np.ndarray):
            side_points = [side_point.tolist() for side_point in side_points]
        obj_string = ''
        for i in range(len(side_points)):
            obj_string += 'v ' + str(side_points[i][0]) + ' ' + str(side_points[i][1]) + ' ' + str(
                side_points[i][2]) + '\n'
        obj_string += f'f '
        for i in range(len(side_points)):
            obj_string += str(global_id + i) + ' '
        obj_string += '\n'
        return obj_string

    @property
    def unit_normal_vector(self):
        """Returns the normal vector of the fracture"""
        if self._unit_normal_vector is None:
            get_side_points = self.get_side_points()
            v1 = get_side_points[1] - get_side_points[0]
            v2 = get_side_points[2] - get_side_points[0]
            cross = np.cross(v1, v2)
            self._unit_normal_vector = cross / np.linalg.norm(cross)
        return self._unit_normal_vector

    def distance_to_point(self, point: np.ndarray):
        """Returns the minimum distance to a point from the fracture plane"""
        a = self.unit_normal_vector[0]
        b = self.unit_normal_vector[1]
        c = self.unit_normal_vector[2]
        d = -np.dot(self.unit_normal_vector, self.centroid)
        return abs(a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

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

    def point_inside_bounding_box(self, point: np.ndarray, scale_factor=0.0):
        """Returns if a point is inside the bounding box of the fracture"""
        bounding_box = self.get_bounding_box()
        lx = bounding_box[1] - bounding_box[0]
        ly = bounding_box[3] - bounding_box[2]
        lz = bounding_box[5] - bounding_box[4]

        if point[0] < bounding_box[0] - scale_factor * lx or point[0] > bounding_box[1] + scale_factor * lx:
            return False
        if point[1] < bounding_box[2] - scale_factor * ly or point[1] > bounding_box[3] + scale_factor * ly:
            return False
        if point[2] < bounding_box[4] - scale_factor * lz or point[2] > bounding_box[5] + scale_factor * lz:
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
        """Returns the plane of the fracture"""
        if not hasattr(self, '_plane'):
            self._plane = Plane(self.centroid, normal=self.unit_normal_vector)

        return self._plane

    @property
    def corners(self) -> List[Point]:
        """Returns the corners of the fracture"""
        if not hasattr(self, '_corners'):
            self._corners = [Point(point) for point in self.get_side_points()]
        return self._corners

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
            Line(self.corners[3], self.corners[0]),
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

    def shift(self, x, y, z):
        """Shifts the fracture"""
        self.x_centroid += x
        self.y_centroid += y
        self.z_centroid += z
        if self._side_points is not None:
            self._side_points = [point + np.array([x, y, z]) for point in self._side_points]

    @property
    def largest_index_normal_vector(self):
        """Returns the largest coordinate index of the normal vector"""
        return np.argmax(self.unit_normal_vector)

    def compute_aperture(self, const=3.020E-3):
        """Computes the aperture of the fracture"""
        if self._aperture is not None:
            return self._aperture
        else:
            # const = config.globals.constants
            # computed_aperture = np.power((12 * const.mu
            #                              * config.globals.constitutive_laws.transmissivity.a
            #                              * np.log10(self.size / 2.0) ** 2) / (const.rho * const.g), 1/3)
            computed_aperture = self.aperture_constant * np.log10(self.size / 2.0)
            return computed_aperture

    @property
    def transmissivity(self):
        """Returns the transmissivity of the fracture"""
        if self._transmissivity is not None:
            return self._transmissivity
        elif self.transmissivity_constant is not None:
            computed_transmissivity = self.transmissivity_constant * (np.log10(self.size / 2.0)) ** 2
            return computed_transmissivity
        else:
            rho = 1000
            g = 9.8
            mu = 8.9E-4
            return (np.power(self.hydraulic_aperture, 3) * rho * g) / (12 * mu)

    @property
    def storativity(self):
        """Returns the storativity of the fracture"""
        if self._storativity is not None:
            return self._storativity
        elif self.storativity_constant is not None:
            computed_storativity = self.storativity_constant * np.log10(self.size / 2.0)
            return computed_storativity
        else:
            return 0.0

    def get_json(self):
        cur_dict = {
            'x': self.x_centroid,
            'y': self.y_centroid,
            'z': self.z_centroid,
            'size': self.size,
            'aperture': self.aperture,
            'dip': self.dip,
            'dip_dir': self.dip_dir,
            'aperture_constant': self.aperture_constant,
            'rock_type': self.rock_type,
            'transmissivity_constant': self.transmissivity_constant,
            'storativity_constant': self.storativity_constant,
            'normal_vector': self._unit_normal_vector.tolist() if self._unit_normal_vector is not None else None,
            'polygon': [point.tolist() for point in self.polygon_points] if self.polygon_points is not None else None,
        }

        return cur_dict




