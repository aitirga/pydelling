import os
import numpy as np
from ..preprocessing.MeshPreprocessor import MeshPreprocessor
import logging
import math
logger = logging.getLogger(__name__)
import meshio as msh
from tqdm import tqdm


class FemReader(MeshPreprocessor):
    upscaled_permeability = {} # dict of upscaled permeability values
    upscaled_porosity = {} # dict of upscaled porosity values

    def __init__(self, filename, kd_tree=True):
        super().__init__()
        self.aux_nodes = []
        self.open_file(filename)
        # Create KD-mesh structure
        if kd_tree:
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


    def _compute_fracture_volume_in_elements(self):
        # Compute volume of fractures in each element.
        #self.elements.total_fracture_volume = np.zeros([len(elements)])
        for elem in tqdm(self.elements, desc="Computing fracture volume fractions"):
            for fracture in elem.associated_fractures:
                fracture_dict = elem.associated_fractures[fracture]
                # Attribute of the element: portion of element occupied by fractures.
                elem.total_fracture_volume += fracture_dict['volume']


    def upscale_mesh_porosity(self, matrix_porosity=None):
        # Compute upscaled porosity for each element.
        self._compute_fracture_volume_in_elements()
        upscaled_porosity = {}
        for elem in tqdm(self.elements, desc="Upscaling porosity"):
            upscaled_porosity[elem.local_id] = (elem.total_fracture_volume / elem.volume)# + (
                       # matrix_porosity[elem] * (1 - (elem.total_fracture_volume / elem.volume)))

        return upscaled_porosity


    def upscale_mesh_permeability(self, matrix_permeability_tensor=None, rho=1000, g=10, mu=10,
                                  mode='full_tensor'):
        # UPSCALED PERMEABILITY
        fracture_perm = {}
        upscaled_perm = {}

        # For each fracture, compute permeability tensor,
        # and add it to the elements intersected by the fracture.
        for elem in tqdm(self.elements, desc="Upscaling permeability"):
            fracture_perm[elem.local_id] = np.zeros([3, 3])
            upscaled_perm[elem.local_id] = np.zeros([3, 3])
            for frac_name in elem.associated_fractures:
                frac_dict = elem.associated_fractures[frac_name]
                frac = frac_dict['fracture']
                perm_tensor = np.zeros([3, 3])
                # n1 = math.cos(frac.dip * (math.pi / 180)) * math.sin(frac.dip_dir * (math.pi / 180))
                # n2 = math.cos(frac.dip * (math.pi / 180)) * math.cos(frac.dip_dir * (math.pi / 180))
                # n3 = -1 * math.sin(frac.dip * (math.pi / 180))
                n1 = frac.unit_normal_vector[0]
                n2 = frac.unit_normal_vector[1]
                n3 = frac.unit_normal_vector[2]
                frac.perm = ((frac.aperture ** 3) * rho * g) / (12 * mu)
                for i in range(1, 4):
                    for j in range(1, 4):
                        perm_tensor[0, 0] = frac.perm * ((n2 ** 2) + (n3 ** 2))
                        perm_tensor[0, 1] = frac.perm * (-1) * n1 * n2
                        perm_tensor[0, 2] = frac.perm * (-1) * n1 * n3
                        perm_tensor[1, 1] = frac.perm * ((n3 ** 2) + (n1 ** 2))
                        perm_tensor[1, 2] = frac.perm * (-1) * n2 * n3
                        perm_tensor[2, 2] = frac.perm * ((n1 ** 2) + (n2 ** 2))

                #Add fracture permeability, weighted by the area that the fracture occupies in the element.
                fracture_perm[elem.local_id][0, 0] += (perm_tensor[0, 0] * frac_dict['volume'] / elem.volume)
                fracture_perm[elem.local_id][0, 1] += (perm_tensor[0, 1] * frac_dict['volume'] / elem.volume)
                fracture_perm[elem.local_id][0, 2] += (perm_tensor[0, 1] * frac_dict['volume'] / elem.volume)
                fracture_perm[elem.local_id][1, 0] += (perm_tensor[0, 1] * frac_dict['volume'] / elem.volume)
                fracture_perm[elem.local_id][1, 1] += (perm_tensor[1, 1] * frac_dict['volume'] / elem.volume)
                fracture_perm[elem.local_id][1, 2] += (perm_tensor[1, 2] * frac_dict['volume'] / elem.volume)
                fracture_perm[elem.local_id][2, 0] += (perm_tensor[0, 1] * frac_dict['volume'] / elem.volume)
                fracture_perm[elem.local_id][2, 1] += (perm_tensor[1, 2] * frac_dict['volume'] / elem.volume)
                fracture_perm[elem.local_id][2, 2] += (perm_tensor[2, 2] * frac_dict['volume'] / elem.volume)

            #Sum permeability contribution from fractures and from matrix.
            upscaled_perm[elem.local_id] = fracture_perm[elem.local_id] #+ matrix_permeability_tensor[elem.local_id] * (
                       # 1 - elem.total_fracture_volume / elem.volume)

        # EXPORT MODES
        # if mode == 'full_tensor':
        #     # mesh_upscaled_perm =
        #     pass
        # elif mode == 'tensor_principals':
        #     eigen_perm = {key: np.zeros([3,3]) for key in elem.ID}
        #     for elem in intersection_dictionary.frac:
        #         eigen_perm = np.linalg.eig(upscaled_perm[elem])
        #     upscaled_perm = eigen_perm
        # else:
        #     print("Isotropic case not implemented yet")

        return upscaled_perm


    def porosity_to_csv(self, filename='./porosity.csv'):
        """Exports porosity values to csv

        Returns:

        """
        import csv
        logger.info(f"Exporting porosity to {filename}")
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for element_id in self.upscaled_porosity:
                centroid = self.elements[element_id].centroid
                porosity = self.upscaled_porosity[element_id]
                writer.writerow([centroid[0], centroid[1], centroid[2], porosity])





