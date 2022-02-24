import os
import numpy as np
from ..preprocessing.MeshPreprocessor import MeshPreprocessor



class FemReader(MeshPreprocessor):
    def __init__(self, filename, **kwargs):
        super().__init__()

    def open_file(self, filename, **kwargs):

        with open(filename, "r") as f:
            for line in f:
                line = line.rstrip()
                split_line = line.split()

                if split_line[0] == "DIMENS":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    self.nodes = int(split_line[0])
                    self.elements = int(split_line[1])

                elif split_line[0] == "VARNODE":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    num_elem_per_node = int(split_line[1])
                    nodes_elem = np.zeros([self.elements, num_elem_per_node])

                    for e in range(0, self.elements):
                        line = f.readline()
                        line = line.rstrip()
                        split_line = line.split()
                        for ne in range(0, num_elem_per_node):
                            nodes_elem[e, ne] = int(split_line[ne + 1])

                elif split_line[0] == "XYZCOOR":
                    line = f.readline()
                    line = line.rstrip()
                    split_line = line.split()
                    for n in range(0, self.nodes):
                        self.add_node(np.array([float(split_line[0][0:-1]), float(split_line[1][0:-1]), float(split_line[2][0:-1])]))
                        line = f.readline()
                        line = line.rstrip()
                        split_line = line.split()
                    break

                else:
                    pass

        for e in range(0, self.elements):
            self.add_tetrahedra(nodes_elem[e], [self.nodes[nodes_elem[e][0]], self.nodes[nodes_elem[e][1]], self.nodes[nodes_elem[e][2]], self.nodes[nodes_elem[e][3]]])
        
