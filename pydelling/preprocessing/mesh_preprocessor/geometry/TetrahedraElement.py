import numpy as np
from scipy.spatial.qhull import ConvexHull

from pydelling.preprocessing.mesh_preprocessor.geometry import TriangleFace, BaseElement


class TetrahedraElement(BaseElement):
    def __init__(self, node_ids, node_coords, centroid_coords=None):
        super().__init__(node_ids=node_ids, node_coords=node_coords, centroid_coords=centroid_coords)
        self.type = "tetrahedra"
        self.meshio_type = "tetra"
        self.define_faces()  # Define faces of the element

        if centroid_coords is None:
            self.centroid = self.compute_centroid()
            self.centroid_coords = self.centroid
        else:
            self.centroid = np.array(centroid_coords)
            self.centroid_coords = self.centroid

    @property
    def volume(self):
        return self.compute_volume()

    def define_faces(self):
        # Add faces that define the wedge
        # Face 1
        self.faces["t1"] = TriangleFace(node_ids=np.array([self.nodes[0],
                                                           self.nodes[1],
                                                           self.nodes[3]]),
                                        node_coords=np.array([self.coords[0],
                                                              self.coords[1],
                                                              self.coords[3]]))

        # Face 2
        self.faces["t2"] = TriangleFace(node_ids=np.array([self.nodes[1],
                                                           self.nodes[2],
                                                           self.nodes[3]]),
                                        node_coords=np.array([self.coords[1],
                                                              self.coords[2],
                                                              self.coords[3]]))
        # Face 3
        self.faces["t3"] = TriangleFace(node_ids=np.array([self.nodes[0],
                                                           self.nodes[3],
                                                           self.nodes[2]]),
                                        node_coords=np.array([self.coords[0],
                                                              self.coords[3],
                                                              self.coords[2]]))
        # Face 4
        self.faces["t4"] = TriangleFace(node_ids=np.array([self.nodes[0],
                                                           self.nodes[2],
                                                           self.nodes[1]]),
                                        node_coords=np.array([self.coords[0],
                                                              self.coords[2],
                                                              self.coords[1]]))

    def compute_volume(self):
        """
        Computes volume of a general polyhedra based on the convex hull of a set of points
        :return: volume of the polyhedron
        """
        return ConvexHull(self.coords, qhull_options='QJ').volume

    def compute_centroid(self):
        """
        Computes the centroid of a general polyhedra
        :return: centroid of the polyhedron
        """
        return np.mean(self.coords, axis=0)







