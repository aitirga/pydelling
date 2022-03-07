import os
import numpy as np
from ..preprocessing.MeshPreprocessor import MeshPreprocessor
import logging
logger = logging.getLogger(__name__)
import meshio as msh
from tqdm import tqdm
from scipy.spatial import KDTree


class FemReader(MeshPreprocessor):
    kd_tree: KDTree
    def __init__(self, filename):
        super().__init__()
        self.aux_nodes = []
        self.open_file(filename)
        # Create KD-mesh structure
        self.create_kd_tree()


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
                    line = line.rstrip().replace(',', '').split()

                    for n in range(0, self.aux_n_nodes):
                        self.aux_nodes.append(np.array([float(line[0]), float(line[1]), float(line[2])]))
                        line = f.readline()
                        line = line.rstrip().replace(',', '').split()
                    break

                else:
                    continue

        for e in range(0, self.aux_n_elements):
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

    def create_kd_tree(self, kd_tree_config=None):
        """
        Create a KD-tree structure for the mesh.
        Args:
            kd_tree_config: A dictionary with the kd-tree configuration.
        """
        if kd_tree_config is None:
            kd_tree_config = {}
        self.kd_tree = KDTree(self.centroids, **kd_tree_config)

    def get_nearest_mesh_elements(self, point, k=15, distance_upper_bound=None):
        """
        Get the nearest mesh elements to a point.
        Args:
            point: A point in 3D space.
            k: The number of nearest elements to return.
        Returns:
            A list of the nearest mesh elements.
        """
        assert hasattr(self, "kd_tree"), "KD-tree not created"
        ids = self.kd_tree.query(point, k=k, distance_upper_bound=None)[1]
        return [self.elements[i] for i in ids]





