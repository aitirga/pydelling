import geometry as geometry
from typing import *
import numpy as np


class MeshPreprocessor(object):
    """Contains the logic to preprocess and work with a generic unstructured mesh"""
    elements: List[geometry.BaseElement]
    nodes: List[np.ndarray]
    centroids: List[np.ndarray]

    def __init__(self):
        self.nodes = []
        self.elements = []
        self.centroids = []

    def add_element(self, element: geometry.BaseElement):
        self.elements.append(element)

    def add_node(self, node: np.ndarray):
        self.nodes.append(node)

    @property
    def n_nodes(self):
        """Returns the number of nodes in the mesh"""
        return len(self.nodes)

    @property
    def n_elements(self):
        """Returns the number of elements in the mesh"""
        return len(self.elements)




