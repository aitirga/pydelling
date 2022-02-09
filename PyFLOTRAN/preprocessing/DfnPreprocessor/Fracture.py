import numpy as np


class Fracture(object):
    def __init__(self, dip, dip_dir, x, y, z, size):
        self.side_points = None
        self.dip = dip
        self.dip_dir = dip_dir
        self.x_centroid = x
        self.y_centroid = y
        self.z_centroid = z
        self.size = size
        self.centroid = np.array([self.x_centroid, self.y_centroid, self.z_centroid])

    def get_side_points_v1(self):
        """
        Finds the side points of the fracture
        Returns: Coordinates of the plane side points
        """
        alpha = (self.dip_dir + 0) / 360 * (2 * np.pi)
        delta = (self.dip + 0) / 360 * (2 * np.pi)

        A = self.centroid + np.array([
            - self.size / 2 * (np.sin(alpha) + np.cos(delta)),
            - self.size / 2 * np.cos(alpha),
            self.size / 2 * np.sin(delta)
        ])

        B = self.centroid + np.array([
            self.size / 2 * (np.sin(alpha) - np.cos(delta)),
            self.size / 2 * np.cos(alpha),
            self.size / 2 * np.sin(delta)
        ])

        C = self.centroid + np.array([
            self.size / 2 * (np.sin(alpha) + np.cos(delta)),
            self.size / 2 * np.cos(alpha),
            - self.size / 2 * np.sin(delta)
        ])

        D = self.centroid + np.array([
            - self.size / 2 * (np.sin(alpha) - np.cos(delta)),
            - self.size / 2 * np.cos(alpha),
            - self.size / 2 * np.sin(delta)
        ])
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
        alpha = self.dip_dir / 360 * (2 * np.pi)
        delta = self.dip / 360 * (2 * np.pi)

        A = self.centroid + np.array([
            + self.size / 2 * (-np.cos(alpha) * np.cos(delta) + np.sin(alpha)),
            + self.size / 2 * (-np.sin(alpha) * np.cos(delta) - np.cos(alpha)),
            self.size / 2 * np.sin(delta)
        ])

        B = self.centroid + np.array([
            + self.size / 2 * (-np.cos(alpha) * np.cos(delta) - np.sin(alpha)),
            + self.size / 2 * (-np.sin(alpha) * np.cos(delta) + np.cos(alpha)),
            self.size / 2 * np.sin(delta)
        ])

        C = self.centroid + np.array([
            + self.size / 2 * (np.cos(alpha) * np.cos(delta) - np.sin(alpha)),
            + self.size / 2 * (np.sin(alpha) * np.cos(delta) + np.cos(alpha)),
            - self.size / 2 * np.sin(delta)
        ])

        D = self.centroid + np.array([
            + self.size / 2 * (np.cos(alpha) * np.cos(delta) + np.sin(alpha)),
            + self.size / 2 * (np.sin(alpha) * np.cos(delta) - np.cos(alpha)),
            - self.size / 2 * np.sin(delta)
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

