import numpy as np
from scipy.spatial.qhull import ConvexHull

from pydelling.readers.iGPReader.geometry import QuadrilateralFace, BaseElement


class HexahedraElement(BaseElement):
    def __init__(self, node_ids, node_coords, element_type_n, local_id, centroid_coords=None):
        super().__init__(node_ids, node_coords, element_type_n, local_id, centroid_coords)
        self.define_faces()
        self.type = "Hexahedra"
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
        self.faces["q1"] = QuadrilateralFace(nodes=np.array([self.nodes[0],
                                                             self.nodes[1],
                                                             self.nodes[5],
                                                             self.nodes[4]]),
                                             coords=np.array([self.coords[0],
                                                              self.coords[1],
                                                              self.coords[5],
                                                              self.coords[4]]))
        # Face 2
        self.faces["q2"] = QuadrilateralFace(nodes=np.array([self.nodes[1],
                                                             self.nodes[2],
                                                             self.nodes[6],
                                                             self.nodes[5]]),
                                             coords=np.array([self.coords[1],
                                                              self.coords[2],
                                                              self.coords[6],
                                                              self.coords[5]]))
        # Face 3
        self.faces["q3"] = QuadrilateralFace(nodes=np.array([self.nodes[2],
                                                             self.nodes[3],
                                                             self.nodes[7],
                                                             self.nodes[6]]),
                                             coords=np.array([self.coords[2],
                                                              self.coords[3],
                                                              self.coords[7],
                                                              self.coords[6]]))
        # Face 4
        self.faces["q4"] = QuadrilateralFace(nodes=np.array([self.nodes[3],
                                                             self.nodes[0],
                                                             self.nodes[4],
                                                             self.nodes[7]]),
                                             coords=np.array([self.coords[3],
                                                              self.coords[0],
                                                              self.coords[4],
                                                              self.coords[7]]))
        # Face 5
        self.faces["q5"] = QuadrilateralFace(nodes=np.array([self.nodes[0],
                                                             self.nodes[3],
                                                             self.nodes[2],
                                                             self.nodes[1]]),
                                             coords=np.array([self.coords[0],
                                                              self.coords[3],
                                                              self.coords[2],
                                                              self.coords[1]]))
        # Face 6
        self.faces["q6"] = QuadrilateralFace(nodes=np.array([self.nodes[4],
                                                             self.nodes[5],
                                                             self.nodes[6],
                                                             self.nodes[7]]),
                                             coords=np.array([self.coords[4],
                                                              self.coords[5],
                                                              self.coords[6],
                                                              self.coords[7]]))

    def compute_volume(self):
        """
        Computes volume of a general polyhedron based on the convex hull of a set of points
        :return: volume of the polyhedron
        """
        return ConvexHull(self.coords).volume

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