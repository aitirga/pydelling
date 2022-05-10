import numpy as np

from pydelling.config import config
from pydelling.readers.iGPReader.io import BaseWriter, iGPReader


class PflotranImplicitWriter(BaseWriter, iGPReader):
    """
    This class contains the function to write a mesh file in PFLOTRAN unstructured implicit format
    """

    def write_elements(self, export_file):
        export_file.write(f"ELEMENTS {self.mesh_info['n_elements']}\n")
        for element in self.elements:
            export_file.write(
                f"{config.globals.element_dict[len(element.coords)]} {' '.join(map(str, np.array(element.nodes) + 1))}\n")

    def write_nodes(self, export_file):
        export_file.write(f"VERTICES {self.mesh_info['n_nodes']}\n")
        for id, node in enumerate(self.nodes_output):
            export_file.write(f"{node[0]} {node[1]} {self.nodes[id][2]:5.5f}\n")

    def write_domain_hdf5(self, export_file):
        """
        Writes the domain info (i.e. grid info) into an hdf5 file
        :param export_file: HDF5 File object
        """
        if self.is_mesh_built:
            raise RuntimeError("Exporting built mesh into hdf5 still under development")
        else:
            #  Pre-process cell data
            cell_data = np.zeros(shape=(self.mesh_info["n_elements"], 9), dtype=np.int32)
            for id, cell in enumerate(self.elements):
                n_cell = len(cell)
                cell_data[id][0] = n_cell
                np.put(cell_data[id], range(1, len(cell) + 1), cell)
            # Pre-process node data
            node_data = np.array(self.nodes)
            domain_group = export_file.create_group("Domain")
            domain_group.create_dataset("Cells", data=cell_data)
            domain_group.create_dataset("Vertices", data=node_data)

    def write_regions_hdf5(self, export_file):
        def add_dataset(hdf5_file, group_name, dataset_type, data):
            temp_group = hdf5_file.create_group(group_name)
            temp_dataset = temp_group.create_dataset(dataset_type, data=data)

        def preprocess_region(dataset):
            """
            Pre-process a face dataset into PFLOTRAN hdf5 format
            :param dataset: input dataset
            :return: pre-processed face dataset
            """
            face_data = np.zeros(shape=(len(dataset), 4), dtype=np.int32)
            for id, face in enumerate(dataset):
                n_face = len(face)
                face_data[id][0] = n_face
                face = np.array(face)
                np.put(face_data[id], range(1, len(face)), face)
            return face_data

        if self.is_mesh_built:
            raise RuntimeError("Exporting built mesh into hdf5 still under development")
        else:
            # Pre-process datasets
            region_group = export_file.create_group("Regions")
            # Add materials
            #  Create 'all' material
            data_all = np.array(range(self.mesh_info["n_elements"])) + 1
            add_dataset(region_group, "all", "Cell Ids", data_all)
            for material_name in self.material_dict:
                add_dataset(region_group, material_name, "Cell Ids", self.material_dict[material_name])
            # Add regions
            for region_name in self.region_dict:
                preprocess_region_dataset = preprocess_region(self.region_dict[region_name]["elements"])
                add_dataset(region_group, region_name, "Vertex Ids", preprocess_region_dataset)
