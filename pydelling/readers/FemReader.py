import numpy as np
from ..preprocessing.mesh_preprocessor import MeshPreprocessor
import logging
import logging

import numpy as np

from ..preprocessing.mesh_preprocessor import MeshPreprocessor

logger = logging.getLogger(__name__)
from tqdm import tqdm


class FemReader(MeshPreprocessor):
    upscaled_permeability = {} # dict of upscaled permeability values
    upscaled_porosity = {} # dict of upscaled porosity values

    def __init__(self, filename, kd_tree=True, st_file=False):
        super().__init__(st_file=False)
        self.aux_nodes = []
        self.open_file(filename)
        # Create KD-mesh structure
        if kd_tree:
            self.create_kd_tree()

    def open_file(self, filename):
        if self.is_streamlit:
            import streamlit as st
        with open(filename, "r") as f:
            for line in f:
                line = line.rstrip()
                split_line = line.split()


                if split_line[0] == "CLASS":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    self.aux_n_layers = int(split_line[4])

                if split_line[0] == "DIMENS":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    self.aux_n_nodes = int(split_line[0])
                    self.aux_n_elements = int(split_line[1])
                    self.aux_num_nodes_per_elem = int(split_line[2])

                elif split_line[0] == "NODE":
                    nodes_elem = np.zeros([self.aux_n_elements, self.aux_num_nodes_per_elem], dtype=int)
                    for e in tqdm(range(0, self.aux_n_elements), desc='Reading elements'):
                        line = f.readline()
                        line = line.rstrip()
                        split_line = line.split()
                        for ne in range(0, self.aux_num_nodes_per_elem):
                            nodes_elem[e, ne] = int(split_line[ne]) - 1

                elif split_line[0] == "VARNODE":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    num_elem_per_node = int(split_line[1])
                    nodes_elem = np.zeros([self.aux_n_elements, num_elem_per_node], dtype=int)
                    element_type = np.zeros([self.aux_n_elements], dtype=int)

                    for e in tqdm(range(0, self.aux_n_elements), ):
                        line = f.readline()
                        line = line.rstrip()
                        split_line = line.split()
                        for ne in range(0, num_elem_per_node):
                            nodes_elem[e, ne] = int(split_line[ne + 1]) - 1
                            element_type[e] = int(split_line[0])


                elif split_line[0] == "COOR":
                    line = f.readline()
                    line = line.rstrip().replace(',', '').split()
                    # for np in tqdm(range(0, self.aux_n_nodes), desc='Reading nodes'):

                    self.aux_xcoor = []
                    self.aux_ycoor = []
                    #self.aux_zcoor = np.zeros([self.aux_n_nodes])  #NOR DEFINITIVE, WE NEED TO READ ELEVATIONS BEFORE CRETING THE NODES.
                    for l in range(0, self.aux_n_nodes/(self.aux_n_layers+1)):
                        for i in range(0,12):
                            self.aux_xcoor.append(float(line[i]))
                    for l in range(0, self.aux_n_nodes/(self.aux_n_layers+1)):
                        for i in range(0,12):
                            self.aux_ycoor.append(float(line[i]))
                    for n in range(0, self.aux_n_nodes):
                        self.aux_nodes.append(np.array([self.aux_xcoor[n], self.aux_ycoor[n], self.aux_zcoor[n]]))
                        #line = f.readline()
                        #line = line.rstrip().replace(',', '').split()
                    break

                elif split_line[0] == "XYZCOOR":
                    line = f.readline()
                    line = line.rstrip().replace(',', '').split()

                    for n in tqdm(range(0, self.aux_n_nodes), desc='Reading nodes'):
                        self.aux_nodes.append(np.array([float(line[0]), float(line[1]), float(line[2])]))
                        line = f.readline()
                        line = line.rstrip().replace(',', '').split()
                    break



                elif split_line[0] == "ELEV_I":
                    line = f.readline()
                    line = line.rstrip().replace(',', '').split()

                    self.aux_elevation_value = []
                    self.nodes_with_this_elevation = []

                    #For each layer:
                        #LOOP TO READ ELEVATION VALUES [Line[0]]
                        #READ ELEVATION VALUE, AND MATRIX WITH MAXIMUM OF 13 COLUMNS HAVING ALL NODES WITH THAT ELEVATION.
                        #                          [Line[1] to Line[14] for I don't know how many lines]

                    break

                else:
                    continue

        if self.is_streamlit:
            st.write(f'Creating {self.aux_n_elements} mesh elements')
            progress_tetra = st.empty()
            progress_tetra.progress(0)
            count = 0


        for e in range(0, self.aux_n_elements):
            if self.is_streamlit:
                proportion = round(count / self.aux_n_elements * 100)
                progress_tetra.progress(proportion)
                count += 1

            if element_type[2] == 6:
                # Add a tetrahedra to the mesh structure
                self.add_tetrahedra(node_ids=nodes_elem[e],
                                    node_coords=[
                                        self.aux_nodes[nodes_elem[e, 0]],
                                        self.aux_nodes[nodes_elem[e, 1]],
                                        self.aux_nodes[nodes_elem[e, 2]],
                                        self.aux_nodes[nodes_elem[e, 3]],
                                    ]
                                    )

            else:
                logger.warning(f"Element type {element_type[e]} not supported")







