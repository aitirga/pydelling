import os
import numpy as np
from ..preprocessing.MeshPreprocessor import MeshPreprocessor



class FemReader(MeshPreprocessor):
    def __init__(self, filename, **kwargs):
        super().__init__()

    def open_file(self, filename, **kwargs):
        
        with open("r", filename) as f:
            for line in f:
                line = line.rstrip()
                split_line = line.split()

                if split_line[0] == "DIMENS":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    self.n_elements = int(split_line[0])
                    self.n_nodes = int(split_line[1])

                    print(line)

                elif split_line[0] == "VARNODE":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    self.n_elem_per_node = int(split_line[1])
                    self.nodes_elem = np.zeros([self.n_node, self.n_elem_per_node])

                    for n in range(0, self.n_node):
                        line = f.readline()
                        line = line.rstrip()
                        split_line = line.split()
                        for ne in range(0, self.n_elem_per_node):
                            self.nodes_elem[n, ne] = int(split_line[ne + 1])

                elif split_line[0] == "XYZCOOR":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    self._mesh_nodes = np.zeros([self.n_elements, 3])
                    for e in range(0, self.n_elements):
                        self._mesh_nodes[e, 0] = float(split_line[0][0:-1])
                        self._mesh_nodes[e, 1] = float(split_line[1][0:-1])
                        self._mesh_nodes[e, 2] = float(split_line[2][0:-1])
                        line = f.readline()
                        line = line.rstrip()
                        split_line = line.split()
                    break

                else:
                    pass
        
