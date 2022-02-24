from typing import *
import numpy as np
import PyFLOTRAN.preprocessing.MeshPreprocessor.geometry as geometry
import meshio as msh


class MeshPreprocessor(object):
    """Contains the logic to preprocess and work with a generic unstructured mesh"""
    elements: List[geometry.BaseElement]
    nodes: List[np.ndarray]
    centroids: List[np.ndarray]
    meshio_mesh: msh.Mesh = None

    def __init__(self):
        self.nodes = []
        self.elements = []
        self.centroids = []

    def add_element(self, element: geometry.BaseElement):
        self.elements.append(element)

    def add_tetrahedra(self, node_ids: List[int], node_coords: List[np.ndarray]):
        self.elements.append(geometry.TetrahedraElement(node_ids=node_ids, node_coords=node_coords))

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

    def convert_mesh_to_meshio(self):
        """Converts the mesh into a meshio mesh
        Returns:
            meshio mesh
        """
        elements_in_meshio = {}
        for element in self.elements:
            if element.type == 'tetrahedra':
                if not 'tetra' in elements_in_meshio.keys():
                    elements_in_meshio['tetra'] = []
                elements_in_meshio['tetra'].append(element.nodes.tolist())
            elif element.type == 'hexahedra':
                if not 'hexahedron' in elements_in_meshio.keys():
                    elements_in_meshio['hexahedron'] = []
                elements_in_meshio['hexahedron'].append(element.nodes.tolist())

        self.meshio_mesh = msh.Mesh(
            points=self.nodes,
            cells=elements_in_meshio
        )

    def to_vtk(self, filename='mesh.vtk'):
        """Converts the mesh into vtk using meshio
        Args:
            filename: name of the output file
        """
        if not self.meshio_mesh:
            self.convert_mesh_to_meshio()
        self.meshio_mesh.write(filename)