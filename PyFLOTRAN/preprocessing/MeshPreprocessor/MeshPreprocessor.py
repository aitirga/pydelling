from typing import *
import numpy as np
import PyFLOTRAN.preprocessing.MeshPreprocessor.geometry as geometry
import meshio as msh
from scipy.spatial import KDTree


class MeshPreprocessor(object):
    """Contains the logic to preprocess and work with a generic unstructured mesh"""
    elements: List[geometry.BaseAbstractMeshObject]
    nodes: List[np.ndarray]
    centroids: List[np.ndarray]
    meshio_mesh: msh.Mesh = None
    kd_tree: KDTree

    def __init__(self):
        self.unordered_nodes = {}
        self.elements = []

    def add_element(self, element: geometry.BaseElement):
        self.elements.append(element)

    def add_tetrahedra(self, node_ids: List[int], node_coords: List[np.ndarray]):
        self.elements.append(geometry.TetrahedraElement(node_ids=node_ids, node_coords=node_coords))
        for idx, node in enumerate(node_coords):
            self.unordered_nodes[node_ids[idx]] = node

    @property
    def nodes(self) -> np.ndarray:
        """Orders the nodes in the mesh"""
        aux_nodes = np.ndarray(shape=(self.n_nodes, 3))
        for idx, node in self.unordered_nodes.items():
            aux_nodes[idx] = node
        return aux_nodes

    def add_quadrilateral(self, node_ids: List[int], node_coords: List[np.ndarray]):
        self.elements.append(geometry.QuadrilateralFace(node_ids=node_ids, node_coords=node_coords))
        for idx, node in enumerate(node_coords):
            self.unordered_nodes[node_ids[idx]] = node

    def add_triangle(self, node_ids: List[int], node_coords: List[np.ndarray]):
        self.elements.append(geometry.TriangleFace(node_ids=node_ids, node_coords=node_coords))
        for idx, node in enumerate(node_coords):
            self.unordered_nodes[node_ids[idx]] = node

    def add_node(self, node: np.ndarray):
        '''Explicitly adds a node (deprecated)'''
        self.nodes.append(node)

    @property
    def n_nodes(self):
        """Returns the number of node_ids in the mesh"""
        return len(self.unordered_nodes)

    @property
    def n_elements(self):
        """Returns the number of elements in the mesh"""
        return len(self.elements)

    def convert_mesh_to_meshio(self):
        """Converts the mesh into a meshio mesh
        Returns:
            meshio mesh
        """

        elements_in_meshio = self._create_meshio_dict(self.elements)
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

    def subset_to_vtk(self, elements: List[geometry.BaseAbstractMeshObject], filename='subset.vtk'):
        """Converts a subset of the mesh into a vtk file
        Args:
            elements: list of element indices
            filename: name of the output file
        """
        assert len(elements) > 0, 'No elements to export'
        subset_mesh = self._convert_subset_to_meshio(elements)
        subset_mesh.write(filename)

    def _convert_subset_to_meshio(self, elements: List[geometry.BaseAbstractMeshObject]) -> msh.Mesh:
        """Converts a subset of the mesh into a meshio mesh
        Args:
            elements: list of element indices
        Returns:
            meshio mesh
        """
        elements_in_meshio = self._create_meshio_dict(elements)
        return msh.Mesh(
            points=self.nodes,
            cells=elements_in_meshio
        )

    def _create_meshio_dict(self, elements: List[geometry.BaseAbstractMeshObject]) -> Dict[str, List[List[int]]]:
        elements_in_meshio = {}
        for element in elements:
            if element.type == 'tetrahedra':
                if not 'tetra' in elements_in_meshio.keys():
                    elements_in_meshio['tetra'] = []
                elements_in_meshio['tetra'].append(element.nodes.tolist())
            elif element.type == 'hexahedra':
                if not 'hexahedron' in elements_in_meshio.keys():
                    elements_in_meshio['hexahedron'] = []
                elements_in_meshio['hexahedron'].append(element.nodes.tolist())
            elif element.type == 'triangle':
                if not 'triangle' in elements_in_meshio.keys():
                    elements_in_meshio['triangle'] = []
                elements_in_meshio['triangle'].append(element.nodes.tolist())
            elif element.type == 'quadrilateral':
                if not 'quad' in elements_in_meshio.keys():
                    elements_in_meshio['quad'] = []
                elements_in_meshio['quad'].append(element.nodes.tolist())

        return elements_in_meshio

    def nodes_to_csv(self, filename='node_ids.csv'):
        """Exports the node_ids to CSV"""
        node_array = np.array(self.nodes)
        np.savetxt(filename, node_array, delimiter=',')

    @property
    def centroids(self):
        """Returns the centroids of the mesh"""
        centroids = []
        for element in self.elements:
            centroids.append(element.centroid)
        return centroids

    def create_kd_tree(self, kd_tree_config=None):
        """
        Create a KD-tree structure for the mesh.
        Args:
            kd_tree_config: A dictionary with the kd-tree configuration.
        """
        if kd_tree_config is None:
            kd_tree_config = {}
        self.kd_tree = KDTree(self.centroids, **kd_tree_config)

    def get_nearest_mesh_elements(self, point, k=15, distance_upper_bound=None):
        """
        Get the nearest mesh elements to a point.
        Args:
            point: A point in 3D space.
            k: The number of nearest elements to return.
        Returns:
            A list of the nearest mesh elements.
        """
        assert hasattr(self, "kd_tree"), "KD-tree not created"
        if distance_upper_bound:
            ids = self.kd_tree.query_ball_point(point, k=k, distance_upper_bound=distance_upper_bound)[1]
        else:
            ids = self.kd_tree.query(point, k=k)[1]

        assert len(ids) != 0, "No elements found"
        return [self.elements[i] for i in ids]



