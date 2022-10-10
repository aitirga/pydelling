import numpy as np
from scipy.spatial.qhull import ConvexHull

from pydelling.readers.iGPReader.geometry import TriangleFace, BaseElement


class TetrahedraElement(BaseElement):
    def __init__(self, node_ids, node_coords, element_type_n, local_id, centroid_coords=None):
        super().__init__(node_ids, node_coords, element_type_n, local_id, centroid_coords)
        self.define_faces()  # Define faces of the element
        self.type = 'Tetrahedra'
        # self.centroid = self.compute_centroid()
        # self.centroid_coords = self.centroid

        if centroid_coords is None:
            self.centroid = self.compute_centroid()
            self.centroid_coords = self.centroid
        else:
            self.centroid = np.array(centroid_coords)
            self.centroid_coords = self.centroid

    def define_faces(self):
        # Add faces that define the wedge
        # Face 1
        self.faces["t1"] = TriangleFace(nodes=np.array([self.nodes[0],
                                                        self.nodes[1],
                                                        self.nodes[3]]),
                                        coords=np.array([self.coords[0],
                                                         self.coords[1],
                                                         self.coords[3]]))

        # Face 2
        self.faces["t2"] = TriangleFace(nodes=np.array([self.nodes[1],
                                                        self.nodes[2],
                                                        self.nodes[3]]),
                                        coords=np.array([self.coords[1],
                                                         self.coords[2],
                                                         self.coords[3]]))
        # Face 3
        self.faces["t3"] = TriangleFace(nodes=np.array([self.nodes[0],
                                                        self.nodes[3],
                                                        self.nodes[2]]),
                                        coords=np.array([self.coords[0],
                                                         self.coords[3],
                                                         self.coords[2]]))
        # Face 4
        self.faces["t4"] = TriangleFace(nodes=np.array([self.nodes[0],
                                                        self.nodes[2],
                                                        self.nodes[1]]),
                                        coords=np.array([self.coords[0],
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

    @property
    def volume(self):
        """Returns the volume of the hexahedra

        Returns: volume of the hexahedra
        """
        return self.compute_volume()