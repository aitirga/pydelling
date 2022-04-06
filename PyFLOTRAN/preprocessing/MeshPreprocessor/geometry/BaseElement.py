import numpy as np

from PyFLOTRAN.config import config
from .BaseFace import BaseFace
import pandas as pd
from .BaseAbstractMeshObject import BaseAbstractMeshObject
from typing import *
from natsort import natsorted
import sympy as sp
from PyFLOTRAN.utils.geometry import Point, Plane, Line
from itertools import product, combinations
from PyFLOTRAN.preprocessing.DfnPreprocessor.Fracture import Fracture


class BaseElement(BaseAbstractMeshObject):
    local_id = 0
    eps = 1E-13

    def __init__(self, node_ids, node_coords, centroid_coords=None):
        self.nodes: np.ndarray = node_ids  # Node id set
        self.coords: np.ndarray = np.array(node_coords)  # Coordinates of each node
        self.centroid_coords = centroid_coords
        self.local_id = BaseElement.local_id # Element id
        BaseElement.local_id += 1
        self.faces: Dict[str, BaseFace] = {}  # Faces of an element
        self.type = None
        self.meshio_type = None
        self.associated_fractures = {}
        self.total_fracture_volume = 00

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

    def intersect_faces_with_plane(self, plane: Plane):
        """Returns the intersection of a face with a plane

        Args:
            face: Face to intersect
            plane: Plane to intersect with

        Returns: Intersection points
        """
        intersected_lines = []
        intersected_points = []

        for face in self.faces:
            intersection = self.faces[face].intersect_with_plane(plane)
            if intersection:
                intersected_lines.append(intersection)


        # Intersect the lines within themselves
        intersected_points = list(self._full_line_intersections(intersected_lines=intersected_lines,
                                                          intersected_points=intersected_points,
                                                          ))
        # Check what happens with the fracture points

        # Check if the intersected points are inside the element
        intersected_inside_points = []
        for point in intersected_points:
            if self.contains(point):
                intersected_inside_points.append(point)
        intersected_points = intersected_inside_points.copy()

        return intersected_points


    def intersect_with_fracture(self, fracture: Fracture):
        """Intersects an element with a fracture"""
        intersected_lines = []
        intersected_points = []

        for face in self.faces:
            intersection = self.faces[face].intersect_with_plane(fracture.plane)
            for corner_line in fracture.corner_lines:
                line_intersection = corner_line.intersect(self.faces[face].plane)
                if line_intersection is not None:
                    if self.contains(line_intersection):
                        intersected_points.append(line_intersection)

            if intersection:
                intersected_lines.append(intersection)

        # Check if the fracture vertex are inside the element
        # for vertex in fracture.corners:
        #     print(self.contains(vertex))


        # Intersect the lines within themselves
        intersected_points = list(self._full_line_intersections(intersected_lines=intersected_lines,
                                                          intersected_points=intersected_points,
                                                          ))
        # Check what happens with the fracture points

        # Check if the intersected points are inside the element
        intersected_inside_points = []
        for point in intersected_points:
            if self.contains(point):
                intersected_inside_points.append(point)
        intersected_points = intersected_inside_points.copy()
        # Test algorithm that moves the corners of the fracture to the closest intersection point
        final_points = []
        for corner in fracture.corners:
            distances = []
            for intersected_point in intersected_points:
                distances.append(corner.distance(intersected_point))
            closest_point = min(distances)
            closest_point_index = distances.index(closest_point)
            final_points.append(intersected_points[closest_point_index])
        # final_points = [np.array([point.x, point.y, point.z]) for point in final_points]

        final_points = np.unique(np.array(final_points), axis=0)
        final_points = [Point(point) for point in final_points]
        # final_points = intersected_points
        return final_points


    def _full_line_intersections(self, intersected_lines: List, intersected_points: List) -> List:
        """Intersects a list of lines with each other"""
        line_combination = list(combinations(intersected_lines, 2))
        for line_pair in line_combination:
            line_1: Line = line_pair[0]
            line_2: Line = line_pair[1]
            intersection = line_1.intersect(line_2)
            if intersection is not None:
                intersected_points.append(intersection)
        return intersected_points

    def contains(self, point: np.ndarray or Point) -> bool:
        """Checks if a point is inside the element"""
        for face in self.faces:
            dot_plane_point = np.dot(self.faces[face].unit_normal_vector, point - self.faces[face].centroid)
            if dot_plane_point > self.eps:
                return False
        return True


    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def edges(self):
        edges_list = []
        for face in self.faces:
            face_edges = self.faces[face].edges
            edges_list.append(face_edges)

        # Delete duplicates
        edge_list_flatten = [sorted(item) for sublist in edges_list for item in sublist]
        edge_list_unique = np.unique(edge_list_flatten, axis=0)
        return edge_list_unique

    @property
    def volume(self):
        """Returns the volume of the hexahedra

        Returns: volume of the hexahedra
        """
        return None

