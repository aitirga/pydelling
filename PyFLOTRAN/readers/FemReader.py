import os
import numpy as np
from ..preprocessing.MeshPreprocessor import MeshPreprocessor
import logging
logger = logging.getLogger(__name__)
import meshio as msh
from tqdm import tqdm


class FemReader(MeshPreprocessor):
    def __init__(self, filename):
        super().__init__()
        self.open_file(filename)

    def open_file(self, filename):
        with open(filename, "r") as f:
            for line in f:
                line = line.rstrip()
                split_line = line.split()

                if split_line[0] == "DIMENS":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    self.aux_n_nodes = int(split_line[0])
                    self.aux_n_elements = int(split_line[1])

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

                elif split_line[0] == "XYZCOOR":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    for n in range(0, self.aux_n_nodes):
                        self.add_node(np.array([float(split_line[0][0:-1]), float(split_line[1][0:-1]), float(split_line[2][0:-1])]))
                        line = f.readline()
                        line = line.rstrip()
                        split_line = line.split()
                    break

                else:
                    pass

        for e in range(0, self.aux_n_elements):
            if element_type[2] == 6:
                # Add a tetrahedra to the mesh structure
                self.add_tetrahedra(node_ids=nodes_elem[e],
                                    node_coords=[
                                        self.nodes[nodes_elem[e, 0]],
                                        self.nodes[nodes_elem[e, 1]],
                                        self.nodes[nodes_elem[e, 2]],
                                        self.nodes[nodes_elem[e, 3]],
                                    ]
                                    )
            else:
                logger.warning(f"Element type {element_type[e]} not supported")








