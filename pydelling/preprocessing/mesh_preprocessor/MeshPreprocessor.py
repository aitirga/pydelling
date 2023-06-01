from __future__ import annotations

import logging
from typing import *

import meshio as msh
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

import pydelling.preprocessing.mesh_preprocessor.geometry as geometry
from pydelling.preprocessing.dfn_preprocessor.Fracture import Fracture
from pydelling.preprocessing.mesh_preprocessor.geometry import BaseElement
from pydelling.utils.geometry_utils import compute_polygon_area

logger = logging.getLogger(__name__)


class MeshPreprocessor(object):
    """Contains the logic to preprocess and work with a generic unstructured mesh"""
    elements: List[geometry.BaseElement]
    coords: List[np.ndarray]
    centroids: List[np.ndarray]
    meshio_mesh: msh.Mesh = None
    kd_tree: KDTree = None
    point_data = {}
    cell_data = {}
    _coords = None
    _centroids = None
    is_intersected = False
    is_streamlit = False
    aux_nodes = {}
    has_kd_tree: bool = False

    def __init__(self, *args, **kwargs):
        self.unordered_nodes = {}
        self.elements = []
        BaseElement.local_id = 0
        if 'st_file' in kwargs:
            self.is_streamlit = True

        self.find_intersection_stats = {
            'total_intersections': 0,
            'intersection_points':
                {
                }
        }

    def add_element(self, element: geometry.BaseElement):
        self.elements.append(element)

    def add_tetrahedra(self, node_ids: List[int] or np.ndarray, node_coords: List[np.ndarray]):
        """Adds a tetrahedron to the mesh"""
        self.elements.append(geometry.TetrahedraElement(node_ids=node_ids, node_coords=node_coords))
        for idx, node in enumerate(node_coords):
            self.unordered_nodes[node_ids[idx]] = node

    def add_hexahedra(self, node_ids: List[int] or np.ndarray, node_coords: List[np.ndarray]):
        """Adds a hexahedron to the mesh"""
        self.elements.append(geometry.HexahedraElement(node_ids=node_ids, node_coords=node_coords))
        for idx, node in enumerate(node_coords):
            self.unordered_nodes[node_ids[idx]] = node

    def add_wedge(self, node_ids: List[int] or np.ndarray, node_coords: List[np.ndarray]):
        """Adds a wedge to the mesh"""
        self.elements.append(geometry.WedgeElement(node_ids=node_ids, node_coords=node_coords))
        for idx, node in enumerate(node_coords):
            self.unordered_nodes[node_ids[idx]] = node

    def add_pyramid(self, node_ids: List[int] or np.ndarray, node_coords: List[np.ndarray]):
        """Adds a pyramid to the mesh"""
        self.elements.append(geometry.PyramidElement(node_ids=node_ids, node_coords=node_coords))
        for idx, node in enumerate(node_coords):
            self.unordered_nodes[node_ids[idx]] = node

    def add_triangular_prism(self, node_ids: List[int] or np.ndarray, node_coords: List[np.ndarray]):
        """Adds a triangular prism to the mesh"""
        self.elements.append(geometry.WedgeElement(node_ids=node_ids, node_coords=node_coords))
        for idx, node in enumerate(node_coords):
            self.unordered_nodes[node_ids[idx]] = node

    @property
    def coords(self) -> np.ndarray:
        """Orders the coords in the mesh"""
        if self._coords is None:
            aux_nodes = np.ndarray(shape=(self.n_nodes, 3))
            for idx, node in self.unordered_nodes.items():
                aux_nodes[idx] = node
            self._coords = aux_nodes
        return self._coords

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
        self.coords.append(node)

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
            points=self.coords,
            cells=elements_in_meshio,
            cell_data=self.cell_data,
            point_data=self.point_data
        )

    def to_vtk(self, filename='mesh.vtk'):
        """Converts the mesh into vtk using meshio
        Args:
            filename: name of the output file
        """
        logger.info(f'Converting mesh to vtk and exporting to {filename}')
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
            points=self.coords,
            cells=elements_in_meshio,
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
            elif element.type == 'wedge':
                if not 'wedge' in elements_in_meshio.keys():
                    elements_in_meshio['wedge'] = []
                elements_in_meshio['wedge'].append(element.nodes.tolist())
            elif element.type == 'triangle':
                if not 'triangle' in elements_in_meshio.keys():
                    elements_in_meshio['triangle'] = []
                elements_in_meshio['triangle'].append(element.nodes.tolist())
            elif element.type == 'quadrilateral':
                if not 'quad' in elements_in_meshio.keys():
                    elements_in_meshio['quad'] = []
                elements_in_meshio['quad'].append(element.nodes.tolist())
            elif element.type == 'pyramid':
                if not 'pyramid' in elements_in_meshio.keys():
                    elements_in_meshio['pyramid'] = []
                elements_in_meshio['pyramid'].append(element.nodes.tolist())

        return elements_in_meshio

    def nodes_to_csv(self, filename='node_ids.csv'):
        """Exports the node_ids to CSV"""
        node_array = np.array(self.coords)
        np.savetxt(filename, node_array, delimiter=',')

    @property
    def centroids(self):
        """Returns the centroids of the mesh"""
        if self._centroids is None:
            centroids = []
            for element in self.elements:
                centroids.append(element.centroid)
            self._centroids = np.array(centroids)
        return self._centroids

    def create_kd_tree(self, kd_tree_config=None):
        """
        Create a KD-tree structure for the mesh.
        Args:
            kd_tree_config: A dictionary with the kd-tree configuration.
        """
        if kd_tree_config is None:
            kd_tree_config = {}
        self.kd_tree = KDTree(self.centroids, **kd_tree_config)

    def get_k_nearest_mesh_elements(self, point, k=15, distance_upper_bound=None):
        """
        Get the nearest mesh elements to a point.
        Args:
            point: A point in 3D space.
            k: The number of nearest elements to return.
        Returns:
            A list of the nearest mesh elements.
        """
        if not hasattr(self, 'kd_tree'):
            self.create_kd_tree()
        if distance_upper_bound:
            ids = self.kd_tree.query(point, k=k, distance_upper_bound=distance_upper_bound)[1]
        else:
            ids = self.kd_tree.query(point, k=k)[1]

        assert len(ids) != 0, "No elements found"
        return [self.elements[i] for i in ids]

    def get_closest_mesh_elements(self, point, distance=None):
        """
        Get the nearest mesh elements to a point inside a distance.
        Args:
            point: A point in 3D space.
            distance: The radius of the sphere.
        Returns:
            A list of the nearest mesh elements.
        """
        if self.kd_tree is None:
            self.create_kd_tree()

        ids = self.kd_tree.query_ball_point(point, distance)
        # assert len(ids) != 0, "No elements found"
        return [self.elements[i] for i in ids]

    def get_closest_n_mesh_elements(self,
                                    point,
                                    n=1
                                    ):
        """
        Get the nearest mesh elements to a point inside a distance.
        Args:
            point: A point in 3D space.
            distance: The radius of the sphere.
        Returns:
            A list of the nearest mesh elements.
        """
        if self.kd_tree is None:
            self.create_kd_tree()

        ids = self.kd_tree.query(point, k=n)[1]
        return [self.elements[i] for i in ids]

    def clear(self):
        """Clears the mesh"""
        self.unordered_nodes = {}
        self.elements = []

    @staticmethod
    def _intersect_fracture_with_element(element, fracture):
        """
        Returns the intersection of a fracture with an element.
        Args:
            element: The element to intersect with.
            fracture: The fracture to intersect with.
        Returns:
            The intersection of the fracture and the element.
        """
        return element.intersect(fracture)

    def _is_fracture_intersected(self, fracture: Fracture, element: geometry.BaseElement):
        """
        Checks if a fracture is intersected by an element.
        Args:
            fracture: The fracture to intersect with.
            element: The element to intersect with.
        Returns:
            True if the fracture is intersected by the element, False otherwise.
        """
        signs = []
        bounding_box: List = fracture.get_bounding_box()
        for coord in element.coords:
            # Check if coord in bounding box
            if bounding_box[0] < coord[0] < bounding_box[1] and bounding_box[2] < coord[1] < bounding_box[3] and \
                    bounding_box[4] < coord[2] < bounding_box[5]:

                distance_to_fracture = fracture.distance_to_point(coord)
                signs.append(np.sign(distance_to_fracture))
            else:
                return False

        if not np.all(np.array(signs) == signs[0]):
            return True


    def find_the_intersection_between_fracture_and_mesh(self, fracture: Fracture):
        """Finds the intersection between a fracture and the mesh"""
        intersections = []
        for element in self.elements:
            if self._is_fracture_intersected(fracture, element):
                intersections.append(element)
        self.subset_to_vtk(intersections, filename='intersections.vtk')

    def find_intersection_points_between_fracture_and_mesh(self, fracture: Fracture, export_stats=False):
        """Finds the intersection points between a fracture and the mesh"""

        intersection_points = []
        kd_tree_filtered_elements = self.get_closest_mesh_elements(fracture.centroid, distance=fracture.size)
        counter = 0
        for element in kd_tree_filtered_elements:
            element: geometry.BaseElement
            counter += 1
            intersection_points = element.intersect_with_fracture(fracture)

            if len(intersection_points) >= 3:
                intersection_area = compute_polygon_area(intersection_points)
                fracture.intersection_dictionary[element.local_id] = intersection_area
                element.associated_fractures[fracture.local_id] = {
                    'area': intersection_area,
                    'volume': intersection_area * fracture.aperture,
                    'fracture': fracture,
                }
            n_intersections = len(intersection_points)
            if not n_intersections in self.find_intersection_stats['intersection_points'].keys():
                self.find_intersection_stats['intersection_points'][n_intersections] = 0
            self.find_intersection_stats['intersection_points'][n_intersections] += 1
            self.find_intersection_stats['total_intersections'] += 1

        self.is_intersected = True

        return intersection_points


    def export_intersection_stats(self, filename='intersection_stats.txt'):
        # Export the run_stats dictionary to file
        assert self.is_intersected, 'The mesh has not been intersected yet.'
        import json
        with open('run_stats.json', 'w') as fp:
            json.dump(self.find_intersection_stats, fp)

    @staticmethod
    def intersect_edge_plane(edge: np.ndarray,
                             edge_point: np.ndarray,
                             plane: Fracture,
                             ) -> np.ndarray or None:
        """This method instersects a given edge with a plane.
        Args:
            edge: The edge to intersect.
            plane: The plane to intersect with.
        Returns:
            The intersection point of the edge and the plane.
        """

        edge_dot = np.dot(edge, plane.unit_normal_vector)
        if edge_dot == 0:
            return None
        else:
            t = -np.dot((edge_point - plane.centroid), plane.unit_normal_vector) / edge_dot
            if t < 1.0 and t > 0.0:
                point = edge_point + t * edge
                if plane.point_inside_bounding_box(point):
                    return point
            else:
                return None

    @property
    def min_x(self):
        return self.coords[:, 0].min()

    @property
    def max_x(self):
        return self.coords[:, 0].max()

    @property
    def min_y(self):
        return self.coords[:, 1].min()

    @property
    def max_y(self):
        return self.coords[:, 1].max()

    @property
    def min_z(self):
        return self.coords[:, 2].min()

    @property
    def max_z(self):
        return self.coords[:, 2].max()


    def save(self, filename):
        """Save the mesh to a file."""
        import pickle
        logger.info(f'Saving mesh to {filename}')
        with open(filename, 'wb') as f:
            save_dictionary = {
                'elements': self.elements,
                'coords': self.coords,
                'kd_tree': self.kd_tree,
                'has_kd_tree': self.has_kd_tree,
            }
            pickle.dump(save_dictionary, f)

    def load(self, filename):
        """Load the mesh from a file."""
        logger.info(f'Loading mesh from {filename}')
        import pickle
        with open(filename, 'rb') as f:
            save_dictionary = pickle.load(f)
            self.elements = save_dictionary['elements']
            self._coords = save_dictionary['coords']
            self.kd_tree = save_dictionary['kd_tree']
            self.has_kd_tree = save_dictionary['has_kd_tree']

    def get_json(self):
        """Export the mesh to a json file."""
        logger.info('Exporting mesh to json')
        save_dictionary = {}
        _elements = [element.get_json() for element in self.elements]
        save_dictionary['elements'] = _elements
        save_dictionary['coords'] = self.coords.tolist()
        save_dictionary['has_kd_tree'] = self.has_kd_tree
        return save_dictionary

    def to_json(self, filename='mesh.json'):
        """Export the mesh to a json file."""
        import json
        with open(filename, 'w') as f:
            json.dump(self.get_json(), f)

    @classmethod
    def from_json(self, filename='mesh.json'):
        """Load the mesh from a json file."""
        logger.info(f'Loading mesh from {filename}')
        BaseElement.local_id = 0
        import json
        with open(filename, 'r') as f:
            save_dictionary = json.load(f)
            mesh = MeshPreprocessor()
            mesh._coords = np.array(save_dictionary['coords'])
            mesh.has_kd_tree = save_dictionary['has_kd_tree']
            MeshPreprocessor.load_elements(mesh, save_dictionary['elements'])
            if mesh.has_kd_tree:
                mesh.create_kd_tree()
            return mesh

    @classmethod
    def from_dict(cls, dict: dict):
        """Load the mesh from a json file."""
        BaseElement.local_id = 0
        mesh = MeshPreprocessor()
        mesh._coords = np.array(dict['coords'])
        mesh.has_kd_tree = dict['has_kd_tree']
        MeshPreprocessor.load_elements(mesh, dict['elements'])
        if mesh.has_kd_tree:
            mesh.create_kd_tree()
        return mesh

    @staticmethod
    def load_elements(mesh: MeshPreprocessor, element_dict):
        """Load the elements from a dictionary."""
        elements = []
        for local_id, element in tqdm(enumerate(element_dict), desc='Loading elements'):
            if element['type'] == 'tetrahedra':
                mesh.add_tetrahedra(node_ids=element['nodes'],
                                    node_coords=mesh._coords[element['nodes']])
            elif element['type'] == 'hexahedra':
                mesh.add_hexahedra(node_ids=element['nodes'],
                                   node_coords=mesh._coords[element['nodes']])
            elif element['type'] == 'wedge':
                mesh.add_wedge(node_ids=element['nodes'],
                               node_coords=mesh._coords[element['nodes']])
            elif element['type'] == 'pyramid':
                mesh.add_pyramid(node_ids=element['nodes'],
                                 node_coords=mesh._coords[element['nodes']])
            associated_fractures_dict = element_dict[local_id]['associated_fractures']
            temp_associated_fractures = {}
            for key in associated_fractures_dict:
                cur_fracture = associated_fractures_dict[key]
                temp_fracture = {key: value for key, value in cur_fracture.items() if key != 'fracture'}
                temp_fracture['fracture'] = cur_fracture['fracture']
                temp_associated_fractures[key] = temp_fracture
            mesh.elements[local_id].associated_fractures = temp_associated_fractures

    def refactor_array_by_element_type(self, array: np.ndarray or list) -> list:
        """Refactors a given array based on the element type"""
        if isinstance(array, np.ndarray):
            array = array.tolist()
        final_array = []
        temp_dict = {
            'wedge': [],
            'pyramid': [],
            'tetrahedra': [],
            'hexahedra': [],
        }
        for element in self.elements:
            temp_dict[element.type].append(array[element.local_id])
        for key in temp_dict:
            if len(temp_dict[key]) > 0:
                final_array.append(temp_dict[key])
        return final_array

    def __repr__(self):
        return f'Mesh with {len(self.elements)} elements and {len(self.coords)} nodes.'

    def add_cell_data(self, name, data):
        self.cell_data[name] = self.refactor_array_by_element_type(data)

    def add_point_data(self, name, data):
        self.point_data[name] = data

