from pydelling.readers.iGPReader.utils.geometry_utils import *
from pydelling.utils.geometry import Plane
from .BaseAbstractMeshObject import BaseAbstractMeshObject


class BaseFace(BaseAbstractMeshObject):
    local_id = 0
    __slots__ = ['node_ids', 'node_coords']
    def __init__(self, node_ids, node_coords):
        self.nodes = np.array(node_ids)
        self.coords = np.array(node_coords)
        self.n_coords = len(node_coords)
        self.type = 'BaseFace'
        # self.local_id = BaseFace.local_id
        # BaseFace.local_id += 1

    @property
    def area(self):
        return self.compute_area()

    @property
    def centroid(self):
        return self.compute_centroid()

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
        c_mean = np.mean(self.coords, axis=0)
        ax.scatter3D(xs=[self.centroid[0]], ys=[self.centroid[1]], zs=[self.centroid[2]], c="r")
        ax.scatter3D(xs=[c_mean[0]], ys=[c_mean[1]], zs=[c_mean[2]], c="g")
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.5))
        ax.axes.set_xlim3d(np.min(self.coords[:, 0]), np.max(self.coords[:, 0]))
        ax.axes.set_ylim3d(np.min(self.coords[:, 1]), np.max(self.coords[:, 1]))
        ax.axes.set_zlim3d(np.min(self.coords[:, 2]), np.max(self.coords[:, 2]))
        plt.show()

    def intersect_with_plane(self, plane: Plane):
        '''Returns the intersection of the face with the plane'''
        return self.plane.intersect(plane)


    @property
    def unit_normal_vector(self):
        if not hasattr(self, '_unit_normal_vector'):
            v1 = self.coords[1] - self.coords[0]
            v2 = self.coords[2] - self.coords[0]
            self._unit_normal_vector = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))

        return self._unit_normal_vector

    @property
    def edges(self):
        '''Returns the edges of the face'''
        return NotImplementedError('This method is not implemented yet')

    @property
    def edge_vectors(self):
        '''Returns the edge vectors of the face'''
        return NotImplementedError('This method is not implemented yet')


    @property
    def plane(self):
        '''Returns the plane of the face'''
        return Plane(self.centroid, normal=self.unit_normal_vector)

    def __repr__(self):
        return f"{self.type}"