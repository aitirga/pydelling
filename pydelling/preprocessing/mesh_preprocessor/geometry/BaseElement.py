from itertools import combinations
from typing import *

import numpy as np

from pydelling.preprocessing.dfn_preprocessor.Fracture import Fracture
from pydelling.utils.geometry import Point, Plane, Line
from pydelling.utils.geometry_utils import filter_unique_points
from .BaseAbstractMeshObject import BaseAbstractMeshObject
from .BaseFace import BaseFace
from scipy.spatial import Delaunay


class BaseElement(BaseAbstractMeshObject):
    local_id = 0
    eps = 1e-2
    eps_zero = 1E-4
    _edge_lines = None
    __slots__ = ['node_ids', 'node_coords']

    def __init__(self, node_ids, node_coords, centroid_coords=None, local_id=None):
        self.nodes: np.ndarray = np.array(node_ids)  # Node id set
        self.coords: np.ndarray = np.array(node_coords)  # Coordinates of each node
        if centroid_coords is None:
            self.centroid = self.compute_centroid()
            self.centroid_coords = self.centroid
        else:
            self.centroid = np.array(centroid_coords)
            self.centroid_coords = self.centroid
        self.local_id = local_id if local_id is not None else BaseElement.local_id
        if local_id is None:
            BaseElement.local_id += 1
        self.faces: Dict[str, BaseFace] = {}  # Faces of an element
        self.type = None
        self.meshio_type = None
        self.associated_fractures = {}
        self.associated_faults = {}
        self.total_fracture_volume = 0
        self.area = 0
        self.is_strange = 0.0

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
        intersected_points = list(self._full_line_intersections(intersected_lines=intersected_lines))
        # Check what happens with the fracture points

        # Check if the intersected points are inside the element
        intersected_inside_points = []
        for point in intersected_points:
            if self.contains(point):
                intersected_inside_points.append(point)
        intersected_points = intersected_inside_points.copy()

        return intersected_points


    def intersect_with_fracture(self, fracture: Fracture, export_all_points=False):
        """Intersects an element with a fracture"""
        intersected_lines = []
        intersected_points = []
        final_points = []
        # First thing, check if the fracture is inside the element
        for corner in fracture.corners:
            if self.contains(corner):
                final_points.append(corner)

        if len(final_points) == fracture.n_side_points:
            final_points = [Point(point) for point in final_points]
            return final_points

        for face in self.faces:
            # intersection = self.faces[face].intersect_with_plane(fracture.plane)
            for corner_line in fracture.corner_lines:
                line_intersection = corner_line.intersect(self.faces[face].plane)
                if line_intersection is not None:
                    # if self.contains(line_intersection):
                    intersected_points.append(line_intersection)
            # if intersection:
            #     intersected_lines.append(intersection)
        # Intersect the element edges with the fracture plane
        for edge in self.edge_lines:
            intersection = edge.intersect(fracture.plane)
            edge_intersections = []
            if export_all_points:
                intersection = edge.intersect(fracture.plane)
                edge_intersections.append(intersection)
                with open('custom_edge_intersections.txt', 'a') as f:
                    import csv
                    writer = csv.writer(f)
                    for point in edge_intersections:
                        writer.writerow([point[0], point[1], point[2]])

            if intersection is not None:
                intersected_points.append(intersection)

        # Intersect the lines within themselves
        # intersected_points += list(self._full_line_intersections(intersected_lines=intersected_lines))
        if export_all_points:
            import csv
            with open('intersected_points_all.csv', 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(intersected_points)
            self.to_obj('element.obj')
        # Check if the intersected points are inside the element
        intersected_inside_points = []
        for point in intersected_points:
            if self.contains(point):
                intersected_inside_points.append(point)

            # if self.on_face(point):
            #     intersected_inside_points.append(point)
        intersected_points = intersected_inside_points.copy()
        if export_all_points:
            with open('intersected_points_inside.csv', 'w') as csvfile:
                import csv
                writer = csv.writer(csvfile)
                writer.writerows(intersected_inside_points)


        # Test algorithm that checks if a point is contained in the fracture
        for point in intersected_points:
            if fracture.contains(point):
                final_points.append(point)
        if export_all_points:
            print(len(final_points))
            print(final_points)
        # Check if any fracture edge is inside the element. If so, add that as intersection point
        # for corner in fracture.corners:
        #     if self.contains(corner):
        #         final_points.append(corner)

        # final_points = intersected_points

        final_points = filter_unique_points(final_points)
        final_points = [Point(point) for point in final_points]

        return final_points

    def _full_line_intersections(self, intersected_lines: List[Line]) -> List:
        """Intersects a list of lines with each other"""
        intersected_points = []
        line_combination = list(combinations(intersected_lines, 2))
        for line_pair in line_combination:
            line_1: Line = line_pair[0]
            line_2: Line = line_pair[1]
            intersection = line_1.intersect(line_2)
            if intersection is not None:
                intersected_points.append(intersection)
        return intersected_points

    def contains(self, point: np.ndarray or Point, sign=1.0) -> bool:
        """Checks if a point is inside the element"""
        # self.plot_normal_vectors()
        # contains = Delaunay(self.coords).find_simplex(point) >= 0
        # return contains
        initial_sign = None
        for face in self.faces:
            face_centroid = self.faces[face].centroid
            vec = point - face_centroid
            norm_vec = vec / np.linalg.norm(vec)
            dot_plane_point = np.dot(self.faces[face].unit_normal_vector, norm_vec)
            # add a tolerance to the dot product
            if abs(dot_plane_point) < self.eps:
                # Set the same sign as initial_sign
                if initial_sign == None:
                    continue
                elif initial_sign == 1.0:
                    dot_plane_point = 1.0
                elif initial_sign == -1.0:
                    dot_plane_point = -1.0
            if initial_sign is None:
                initial_sign = np.sign(dot_plane_point)
                continue
            if np.sign(dot_plane_point) != initial_sign:
                # self.plot_normal_vectors(point=point, value=dot_plane_point, error_face=face_centroid)
                return False
        return True

    def on_face(self, point):
        """Checks if a point is on a face of the element"""
        for face in self.faces:
            face_centroid = self.faces[face].centroid
            vec = point - face_centroid
            norm_vec = vec / np.linalg.norm(vec)
            dot_plane_point = np.abs(np.dot(self.faces[face].unit_normal_vector, norm_vec))
            if dot_plane_point < self.eps_zero:
                return True
        return False

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

    def get_json(self):
        """Returns a json representation of the element"""
        serialized_associated_fractures = {key: {
            'area': value['area'],
            'volume': value['volume'],
            'fracture': value['fracture'],
        } for key, value in self.associated_fractures.items()}



        json_dict = {
            "type": self.type,
            "local_id": self.local_id,
            "n_nodes": self.n_nodes,
            "nodes": self.nodes.tolist(),
            "associated_fractures": serialized_associated_fractures,
            "associated_faults": self.associated_faults,
            "total_fracture_volume": self.total_fracture_volume,
            "area": self.area,
        }
        return json_dict

    @property
    def edge_lines(self):
        if self._edge_lines is None:
            # Compute the edge lines
            all_edges = []
            all_lines = []
            for face in self.faces:
                face_edges = self.faces[face].edge_vectors
                for edge_id, edge in enumerate(face_edges):
                    if not self.arr_in_seq(-edge, all_edges):
                        all_edges.append(edge)
                        all_lines.append(Line(p1=self.faces[face].coords[edge_id], direction_vector=edge))
            self._edge_lines = all_lines
        return self._edge_lines


    @staticmethod
    def arr_in_seq(arr, seq):
        tp = type(arr)
        return any(isinstance(e, tp) and np.array_equiv(e, arr) for e in seq)

    @property
    def local_face_nodes(self) -> dict:
        """Returns the local node ids for each face"""
        return {}

    def to_obj(self, filename: str):
        """Exports the element to an obj file"""
        with open(filename, 'w') as f:
            f.write('# OBJ file\n')
            f.write('# Created by pydelling\n')
            f.write('# vertices\n')
            for coord in self.coords:
                f.write('v {} {} {}\n'.format(coord[0], coord[1], coord[2]))
            f.write('# faces\n')
            for face_name in self.local_face_nodes:
                local_ids = self.local_face_nodes[face_name]
                f.write('f ')
                for local_id in local_ids:
                    f.write(f'{local_id + 1} ')
                f.write('\n')


    def compute_centroid(self):
        """
        Computes the centroid of a general polyhedra
        :return: centroid of the polyhedron
        """
        return np.mean(self.coords, axis=0)

    def plot_normal_vectors(self, point: Point=None, value=None, error_face=None):
        """Plots the normal vectors of the faces"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for face in self.faces:
            face_centroid = self.faces[face].centroid
            ax.quiver(face_centroid[0], face_centroid[1], face_centroid[2], self.faces[face].unit_normal_vector[0],
                      self.faces[face].unit_normal_vector[1], self.faces[face].unit_normal_vector[2], length=0.5,
                      normalize=True)
        # Add transparent faces
        for face in self.faces:
            face_coords = self.faces[face].coords
            face_coords = np.array(face_coords)
            ax.add_collection3d(
                Poly3DCollection([face_coords], facecolors='b', alpha=0.1, linewidths=1, edgecolors='k'))


        if point is not None:
            ax.scatter(point[0], point[1], point[2], color='r')

        if value is not None:
            ax.text(point[0], point[1], point[2], value)

        if error_face is not None:
            ax.scatter(error_face[0], error_face[1], error_face[2], color='g')
        # Set aspect ratio
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

        plt.show()
