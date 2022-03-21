import numpy as np


class Fracture(object):
    local_id = 0
    def __init__(self, dip, dip_dir, x, y, z, size, aperture=0.001):
        self.side_points = None
        self.dip = dip
        self.dip_dir = dip_dir
        self.x_centroid = x
        self.y_centroid = y
        self.z_centroid = z
        self.size = size
        self.centroid = np.array([self.x_centroid, self.y_centroid, self.z_centroid])
        self.intersection_dictionary = {}
        self.aperture = aperture
        self.local_id = Fracture.local_id
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



