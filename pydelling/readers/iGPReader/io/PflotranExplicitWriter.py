import os

from pydelling.config import config
from pydelling.readers.iGPReader.geometry import *
from pydelling.readers.iGPReader.io import iGPReader
from pydelling.readers.iGPReader.utils.geometry_utils import *


class PflotranExplicitWriter(iGPReader):
    """
    This class contains the functions to write a mesh file in PFLOTRAN unstructured explicit format
    """
    def write_cells(self, export_file):
        export_file.write(f"CELLS {len(self.elements)}\n")
        for element in self.elements:
            export_file.write(
                f"{element.local_id + 1} {element.centroid[0]:1.8e} {element.centroid[1]:1.8e} {element.centroid[2]:1.8e} {element.volume:1.8e}\n")

    def write_connections(self, export_file):
        # compute number of connection elements
        n_conn = 0
        for element in self.connections:
            n_conn += len(element[1])
        self.n_conn = n_conn  # set the number of connections of the mesh
        export_file.write(f"CONNECTIONS {self.n_conn}\n")
        for element in self.connections:
            prime_element = element[0]
            for connected_element in element[1]:
                face_id = element[1][connected_element]
                face_obj = self.elements[prime_element].faces[face_id]
                face_nodes = face_obj.coords
                # Compute line-face intersection
                line_points = np.array([self.elements[prime_element].centroid_coords,
                                        self.elements[connected_element].centroid_coords])
                intersection_point = line_plane_intersection(line_points=line_points,
                                                             plane_points=face_nodes)
                # compute connection centroid and area
                # intersection_point = face_obj.centroid
                conn_area = face_obj.area
                # export_file.write(f"{prime_element + 1} {connected_element + 1} {conn_centroid[0]:1.4e} {conn_centroid[1]:1.4e} {conn_centroid[2]:1.4e} {conn_area:1.4e}\n")
                export_file.write(
                    f"{prime_element + 1} {connected_element + 1} {intersection_point[0]:1.8e} {intersection_point[1]:1.8e} {intersection_point[2]:1.8e} {conn_area:1.8e}\n")

    def write_condition_data(self):
        for condition in self.region_dict:
            # Open file for dumping connection data
            if self.output_folder is None:
                file_cond = open(f"{condition}.ex", "w")
            else:
                file_cond = open(os.path.join(self.output_folder, f"{condition}.ex"), "w")
            file_cond.write(f"CONNECTIONS {self.region_dict[condition]['length']}\n")
            for id_number, element in enumerate(self.region_dict[condition]['elements']):
                conn_coords = self.nodes[element]
                if len(conn_coords == 3):
                    conn_face = TriangleFace(nodes=element,
                                             coords=conn_coords)
                elif len(conn_coords == 4):
                    conn_face = QuadrilateralFace(nodes=element,
                                                  coords=conn_coords)
                conn_area = conn_face.area
                conn_element_id = self.region_dict[condition]["centroid_id"][id_number]
                line_point = self.elements[conn_element_id].centroid_coords
                intersection_point = line_plane_perpendicular_intersection(line_point=line_point,
                                                                           plane_points=conn_coords)
                file_cond.write(
                    f"{conn_element_id + 1} {intersection_point[0]:1.8e} {intersection_point[1]:1.8e} {intersection_point[2]:1.8e} {conn_area:1.8e}\n")
            file_cond.close()

    def write_domain_postprocess_hdf5(self, export_file):

        # Create domain group
        domain_group = export_file.create_group("Domain")
        # Create Cells dataset
        _cells = []
        for element in self.elements:
            element_length = len(element.coords)
            _cells.append(config.globals.explicit_writer_dict[element_length])
            # Add elements on the _cells dataset
            list(map(lambda x: _cells.append(x), element.nodes))
        domain_group.create_dataset("Cells", data=_cells)
        # Create Vertices
        domain_group.create_dataset("Vertices", data=self.nodes)

    def write_materials(self):
        for material_name in self.material_dict:
            if self.output_folder is None:
                file_mat = open(f"{material_name}.mat", "w")
            else:
                file_mat = open(os.path.join(self.output_folder, f"{material_name}.mat"), "w")
            # Write material id elements
            list(map(lambda x: file_mat.write(f"{x}\n"), self.material_dict[material_name]))
            file_mat.close()
