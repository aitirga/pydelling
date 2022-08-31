import os
from pathlib import Path

from pydelling.readers.iGPReader.io import iGPReader
from pydelling.readers.iGPReader.utils.geometry_utils import *


class CsvWriter(iGPReader):
    def write_csv_cells(self):
        if not self.output_folder:
            file_csv = open(f"{self.project_name}_cell.csv", "w")
        else:
            file_csv = open(Path.cwd() / f"output/{self.project_name}_cell.csv", "w")
        file_csv.write("X,Y,Z,V\n")
        for element in self.elements:
            file_csv.write(f"{element.centroid[0]},{element.centroid[1]},{element.centroid[2]},{element.volume}\n")
        file_csv.close()

    def write_csv_connection(self):
        if self.output_folder is None:
            file_csv = open(f"{self.project_name}_connections.csv", "w")
        else:
            file_csv = open(os.path.join(self.output_folder, f"{self.project_name}_connections.csv"), "w")
        file_csv.write("X,Y,Z,A\n")
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
                conn_area = face_obj.area
                conn_centroid = face_obj.centroid
                # export_file.write(f"{prime_element + 1} {connected_element + 1} {conn_centroid[0]:1.4e} {conn_centroid[1]:1.4e} {conn_centroid[2]:1.4e} {conn_area:1.4e}\n")
                file_csv.write(f"{intersection_point[0]},{intersection_point[1]},{intersection_point[2]},{conn_area}\n")
        file_csv.close()
