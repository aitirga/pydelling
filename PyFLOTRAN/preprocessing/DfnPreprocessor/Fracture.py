import numpy as np


class Fracture(object):
    def __init__(self, dip, dip_dir, x, y, z, size):
        self.dip = dip
        self.dip_dir = dip_dir
        self.x_centroid = x
        self.y_centroid = y
        self.z_centroid = z
        self.size = size
        self.centroid = np.array([self.x_centroid, self.y_centroid, self.z_centroid])

    def get_side_points(self):
        """
        Finds the side points of the fracture
        Returns: Coordinates of the plane side points
        """
        alpha = self.dip_dir / 360 * (2 * np.pi)
        delta = self.dip / 360 * (2 * np.pi)

        A = self.centroid + np.array([
            - self.size / 2 * (np.sin(alpha) + np.cos(alpha)),
            - self.size / 2 * np.cos(alpha),
            self.size / 2 * np.sin(delta)
        ])

        B = self.centroid + np.array([
            self.size / 2 * (np.sin(alpha) - np.cos(alpha)),
            self.size / 2 * np.cos(alpha),
            self.size / 2 * np.sin(delta)
        ])

        C = self.centroid + np.array([
            self.size / 2 * (np.sin(alpha) + np.cos(alpha)),
            self.size / 2 * np.cos(alpha),
            - self.size / 2 * np.sin(delta)
        ])

        D = self.centroid + np.array([
            - self.size / 2 * (np.sin(alpha) - np.cos(alpha)),
            - self.size / 2 * np.cos(alpha),
            - self.size / 2 * np.sin(delta)
        ])

        return np.array([A, B, C, D])



    def get_side_points_v1(self):
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



