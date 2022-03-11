import numpy as np

from PyFLOTRAN.config import config
from .BaseFace import BaseFace
from .BaseAbstractMeshObject import BaseAbstractMeshObject
from typing import *


class BaseElement(BaseAbstractMeshObject):
    local_id = 0
    def __init__(self, node_ids, node_coords, centroid_coords=None):
        self.nodes: np.ndarray = node_ids  # Node id set
        self.coords: np.ndarray = np.array(node_coords)  # Coordinates of each node
        self.centroid_coords = centroid_coords
        self.local_id = BaseElement.local_id # Element id
        BaseElement.local_id += 1
        self.faces: Dict[str, BaseFace] = {}  # Faces of an element
        self.type = None
        self.meshio_type = None

    def __repr__(self):
        return f"{self.type} {self.local_id}"

    def print_element_info(self):
        print("### Element info ###")
        print(f"Element ID: {self.local_id}")
        print(f"Number of nodes: {self.n_nodes}")
        print(f"Element type: {self.type}")
        print(f"Node list: {self.nodes}")
        print("### End element info ###")

    def print_face_info(self):
        print("### Face info ###")
        for face in self.faces:
            print(f"{face}: {self.faces[face].nodes}")
        print("### End face info ###")

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def edges(self):
        edges_list = []
        for face in self.faces:
            edges_list.append(self.faces[face].edges)
        return edges_list

    @property
    def edge_vectors(self):
        edges_vectors_list = []
        for face in self.faces:
            edges_vectors_list.append(self.faces[face].edge_vectors)
        return edges_vectors_list
