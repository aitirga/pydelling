from pydelling.readers.iGPReader.utils.geometry_utils import *


class BaseFace:
    def __init__(self, nodes, coords):
        self.nodes = nodes
        self.coords = coords
        self.n_coords = len(coords)
        self.area = self.compute_area()
        self.centroid = self.compute_centroid()
        self.type = "BaseFace"
        # print(f"old centroid: {self.compute_centroid_mean()}, new centroid: {self.centroid}")
        # print(f"Area: {self.area}")
        # if len(self.node_ids) == 3:
        #     self.plot_face()

    def compute_area(self):
        """
        This function computes the area of a 3D planar polygon
        :return: Area of a 3D planar polygon
        """
        assert len(self.coords >= 3), "Incorrect number of points, more are needed to form a polygon"
        # Compute normal
        vn = normal_vector(self.coords)
        temp_area_v = np.zeros(shape=3)
        projected_area = 0.0
        for id, point in enumerate(self.coords):
            # Set-up variables
            v1 = self.coords[id % self.n_coords]  # Pv1, assuming P=(0,0,0)
            v2 = self.coords[(id + 1) % self.n_coords]  # Pv2, assuming P=(0,0,0)
            # Compute area
            id_area_v = np.cross(v1, v2)
            projected_area_id = np.dot(vn, id_area_v) / 2.0  # area of small triangle of the face
            projected_area += projected_area_id
        return projected_area

    def compute_centroid(self):
        return np.mean(self.coords, axis=0)

    def compute_centroid_mean(self):
        return np.mean(self.coords, axis=0)

    def plot_face(self):
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt
        x = []
        y = []
        z = []
        # points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
        for point in self.coords:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        fig = plt.figure()
        ax = Axes3D(fig)
        verts = [list(zip(x, y, z))]
        # verts = list(zip(self.node_coords))
        c_mean = np.mean(self.coords, axis=0)
        ax.scatter3D(xs=[self.centroid[0]], ys=[self.centroid[1]], zs=[self.centroid[2]], c="r")
        ax.scatter3D(xs=[c_mean[0]], ys=[c_mean[1]], zs=[c_mean[2]], c="g")
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.5))
        ax.axes.set_xlim3d(np.min(self.coords[:, 0]), np.max(self.coords[:, 0]))
        ax.axes.set_ylim3d(np.min(self.coords[:, 1]), np.max(self.coords[:, 1]))
        ax.axes.set_zlim3d(np.min(self.coords[:, 2]), np.max(self.coords[:, 2]))
        plt.show()
