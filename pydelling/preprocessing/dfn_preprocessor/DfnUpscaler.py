import logging
import pathlib

import dill
import numpy as np
import math
from tqdm import tqdm

import pydelling.preprocessing.mesh_preprocessor.geometry as geometry
import pydelling.preprocessing.mesh_preprocessor.geometry.BaseElement
from pydelling.preprocessing.dfn_preprocessor import DfnPreprocessor
from pydelling.preprocessing.dfn_preprocessor import Fracture, Fault
from pydelling.preprocessing.mesh_preprocessor import MeshPreprocessor
from pydelling.utils.geometry_utils import compute_polygon_area

logger = logging.getLogger(__name__)


class DfnUpscaler:
    def __init__(self, dfn: DfnPreprocessor,
                 mesh: MeshPreprocessor,
                 parallel=False,
                 save_intersections=False,
                 load_faults:str or pathlib.Path=None,
                 loading=False,
                 nearest=None,
                 check_nodes=False,
                 ):
        self.eps = 1E-4
        self.dfn: DfnPreprocessor = dfn
        self.mesh: MeshPreprocessor = mesh
        logger.info('The DFN and mesh objects have been set properly')
        self.all_intersected_points = []
        self.save_intersections = save_intersections
        self.load_faults = load_faults

        self.target_num = 15
        self.cur_num = 0
        self.add_to_class('nearest', nearest, default=None)
        self.add_to_class('check_nodes', check_nodes, default=False)

        if not loading:
            self._intersect_dfn_with_mesh(parallel=parallel)



    def _intersect_dfn_with_mesh(self, parallel=False):
        """Runs the DfnUpscaler"""
        logger.info('Upscaling the DFN to the mesh')
        self.break_for = False
        for fracture in tqdm(self.dfn, desc='Intersecting fractures with mesh', total=len(self.dfn)):
            self.find_intersection_points_between_fracture_and_mesh(fracture)
            # if fracture.local_id == 94:
            #     break
            # if self.break_for:
            #     break

        if not self.load_faults:
            self.find_fault_cells(nearest=self.nearest, check_nodes=self.check_nodes)
        else:
            logger.info(f'Loading fault assignment information from {self.load_faults}')
            with open(self.load_faults, 'rb') as f:
                fault_info = dill.load(f)
                for element in self.mesh.elements:
                    element.associated_faults = fault_info[element.local_id]


        if self.save_intersections:
            import csv
            with open('intersections.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'z', 'value'])
                local_id = 0
                for intersection in self.all_intersected_points:
                    for point in intersection:
                        writer.writerow([point.x, point.y, point.z, local_id])
                    local_id += 1

    def find_intersection_points_between_fracture_and_mesh(self, fracture: Fracture, export_stats=True):
        """Finds the intersection points between a fracture and the mesh"""

        intersection_points = []
        kd_tree_filtered_elements = self.mesh.get_closest_mesh_elements(fracture.centroid, distance=fracture.size)
        counter = 0
        elements_filtered = 0
        for element in kd_tree_filtered_elements:
            element: geometry.BaseElement
            counter += 1
            absolute_distance = np.abs(fracture.distance_to_point(element.centroid))
            characteristic_length = np.power(element.volume, 1 / 3)
            if absolute_distance > 1.25 * characteristic_length:
                elements_filtered += 1
                continue

            intersection_points = element.intersect_with_fracture(fracture)
            if len(intersection_points) > 0:
                a = 1
            if self.save_intersections:
                self.all_intersected_points.append(intersection_points)

            # intersection_area = compute_polygon_area(intersection_points)
            intersection_area = np.abs(compute_polygon_area(intersection_points))
            fracture.intersection_dictionary[element.local_id] = intersection_area
            if len(intersection_points) > 0:
                element.associated_fractures[fracture.local_id] = {
                    'area': intersection_area,
                    'volume': intersection_area * fracture.aperture,
                    'fracture': fracture.local_id,
                }
            n_intersections = len(intersection_points)
            if not n_intersections in self.mesh.find_intersection_stats['intersection_points'].keys():
                self.mesh.find_intersection_stats['intersection_points'][n_intersections] = 0
            self.mesh.find_intersection_stats['intersection_points'][n_intersections] += 1
            self.mesh.find_intersection_stats['total_intersections'] += 1

        self.mesh.is_intersected = True
        return intersection_points

    def find_fault_cells(self, save_fault_cells=True,
                         nearest=None,
                         check_nodes=False,
                         ):
        """Finds the fault cells in the mesh"""
        logger.info('Finding fault cells')
        fault_cells = {}
        for fault in tqdm(self.dfn.faults, desc='Finding distances to faults'):
            fault: Fault
            # Iterate over each triangle individually and find close mesh elements
            triangle_centers = fault.trimesh_mesh.triangles_center
            triangle_areas = fault.trimesh_mesh.area_faces
            characteristic_distance = fault.effective_aperture if fault.effective_aperture is not None else fault.aperture
            logger.info(f'Processing fault {fault.local_id} containing {len(triangle_centers)} triangles')
            close_triangles = []
            for triangle_center in triangle_centers:
                if not nearest:
                    kd_tree_filtered_elements = self.mesh.get_closest_mesh_elements(triangle_center, distance=characteristic_distance)
                    if len(kd_tree_filtered_elements) == 0:
                        continue
                else:
                    nearest: int
                    assert type(nearest) == int, 'nearest must be an integer'
                    kd_tree_filtered_elements = self.mesh.get_closest_n_mesh_elements(triangle_center, n=nearest)
                close_triangles.append(kd_tree_filtered_elements)
            close_triangles = [item for sublist in close_triangles for item in sublist]
            # Filter out duplicates
            temp_dict = {triangle.local_id: triangle for triangle in close_triangles}
            close_triangles = list(temp_dict.values())

            kd_tree_centroids = np.array([elem.centroid for elem in close_triangles])
            # Add element nodes


            logger.info(f'Found {len(kd_tree_centroids)} close elements, computing distances to mesh.')
            distances = fault.distance(kd_tree_centroids)

            for element, distance in zip(close_triangles, distances):
                distance = np.abs(distance)
                if not nearest:
                    if distance < fault.aperture / 2:
                        element.associated_faults[fault.local_id] = {
                            'distance': distance,
                        }
                        fault.associated_elements.append(element)

                else:
                    element.associated_faults[fault.local_id] = {
                        'distance': distance if distance > self.eps else self.eps,
                    }
                    fault.associated_elements.append(element)

            # Check node distances
            if check_nodes:
                logger.info('Checking node distances')
                number_of_nodes = []
                all_nodes = []
                for element in close_triangles:
                    element: pydelling.preprocessing.mesh_preprocessor.geometry.BaseElement
                    nodes = element.coords
                    number_of_nodes.append(len(nodes))
                    all_nodes += [node for node in nodes]
                # all_nodes = np.array(all_nodes).reshape(-1, 3)
                distances = fault.distance(np.array(all_nodes))
                reconstructed_distances = []
                cum_idx = 0
                for n_node in number_of_nodes:
                    reconstructed_distances.append(distances[cum_idx:cum_idx + n_node])
                    cum_idx += n_node

                for element_id, element in enumerate(close_triangles):
                    element: pydelling.preprocessing.mesh_preprocessor.geometry.BaseElement
                    distances = reconstructed_distances[element_id]
                    for distance in distances:
                        if distance < fault.aperture / 2:
                            element.associated_faults[fault.local_id] = {
                                'distance': distance if distance > self.eps else self.eps,
                            }
                            fault.associated_elements.append(element)
                            break

        if save_fault_cells:
            import dill
            for element in self.mesh.elements:
                fault_cells[element.local_id] = element.associated_faults
            with open('fault_cells.pkl', 'wb') as f:
                dill.dump(fault_cells, f)




    def _compute_fracture_volume_in_elements(self):
        # Compute volume of fractures in each element.
        # self.elements.total_fracture_volume = np.zeros([len(elements)])
        for elem in tqdm(self.mesh.elements, desc="Computing fracture volume fractions"):
            for fracture in elem.associated_fractures:
                fracture_dict = elem.associated_fractures[fracture]
                # Attribute of the element: portion of element occupied by fractures.
                elem.total_fracture_volume += fracture_dict['volume']

    def upscale_mesh_porosity(self,
                              matrix_porosity=None,
                              intensity_correction_factor=1.0,
                              existing_fractures_fraction=1.0,
                              truncate_to_min_percentile=5,
                              truncate_to_max_percentile=95,
                              truncate=True,
                              ):
        # Compute upscaled porosity for each element.
        matrix_porosity = 0.0
        self._compute_fracture_volume_in_elements()
        upscaled_porosity = {}
        for elem in tqdm(self.mesh.elements, desc="Upscaling porosity"):
            element_volume = elem.volume
            upscaled_porosity[elem.local_id] = (elem.total_fracture_volume / element_volume) + matrix_porosity * (1 - (elem.total_fracture_volume / element_volume))
            upscaled_porosity[elem.local_id] = np.abs(upscaled_porosity[elem.local_id]) * intensity_correction_factor * (1 / existing_fractures_fraction)

        for fault in self.dfn.faults:
            for element in fault.associated_elements:
                upscaled_porosity[element.local_id] = fault.porosity

        #Post-processing: Truncate values to P5 and P95.
        if truncate:
            resulting_porosity = [upscaled_porosity[local_id] for local_id in upscaled_porosity]
            minimum_porosity = -1.0
            min_percentile = truncate_to_min_percentile - 1
            while minimum_porosity <= 0:
                minimum_porosity = np.percentile(np.array(resulting_porosity)[~np.isnan(resulting_porosity)],
                                                 min_percentile)
                min_percentile = min_percentile + 1
            print("Porosity will be truncated to Percentile = " + str(min_percentile))

            # Truncate to max
            maximum_porosity = np.percentile(np.array(resulting_porosity)[~np.isnan(resulting_porosity)],
                                                truncate_to_max_percentile)  # TODO: check if this is correct (strange results with small datasets)
            maximum_porosity = np.max(np.array(resulting_porosity)[~np.isnan(resulting_porosity)])

            for elem in tqdm(self.mesh.elements, desc="Truncating porosity values to P1 and P99"):

                if math.isnan(upscaled_porosity[elem.local_id]):
                    upscaled_porosity[elem.local_id] = minimum_porosity
                elif upscaled_porosity[elem.local_id] > maximum_porosity:
                    upscaled_porosity[elem.local_id] = maximum_porosity
                elif upscaled_porosity[elem.local_id] <= 0.0:
                    upscaled_porosity[elem.local_id] = minimum_porosity
                else:
                    continue

        vtk_porosity = np.asarray(self.mesh.elements)
        for local_id in upscaled_porosity:
            vtk_porosity[local_id] = upscaled_porosity[local_id]

        refactored_porosity = self.mesh.refactor_array_by_element_type(vtk_porosity)

        self.mesh.cell_data['upscaled_porosity'] = refactored_porosity
        self.upscaled_porosity = upscaled_porosity

        return upscaled_porosity

    def upscale_mesh_storativity(self,
                                 matrix_storativity=None,
                                 truncate_to_min_percentile=5,
                                 truncate_to_max_percentile=95,
                                 truncate=True,
                                 ):

        upscaled_storativity = {}

        for elem in tqdm(self.mesh.elements, desc="Upscaling fractures storativity"):
            upscaled_storativity[elem.local_id] = 0.0
            element_volume = elem.volume
            sum_weights = 0.0
            sum_weigted_storativity = 0.0
            for frac_name in elem.associated_fractures:
                frac_dict = elem.associated_fractures[frac_name]
                frac = frac_dict['fracture']
                frac_volume_in_element = frac_dict['volume'] / element_volume
                sum_weights += frac_volume_in_element
                sum_weigted_storativity += self.dfn[frac].storativity * frac_volume_in_element
                upscaled_storativity[elem.local_id] = sum_weigted_storativity/sum_weights


        for fault in self.dfn.faults:
            for element in fault.associated_elements:
                upscaled_storativity[element.local_id] = fault.storativity


        #Post-processing: Truncate values to P5 and P95.
        if truncate:
            resulting_storativity = [upscaled_storativity[local_id] for local_id in upscaled_storativity]
            minimum_storativity = -1.0
            min_percentile = truncate_to_min_percentile - 1
            while minimum_storativity <= 0:
                minimum_storativity = np.percentile(np.array(resulting_storativity)[~np.isnan(resulting_storativity)], min_percentile)
                min_percentile = min_percentile + 1
                print(min_percentile)

            print("Storativity will be truncated to Percentile = " + str(min_percentile))

            maximum_storativity = np.percentile(np.array(resulting_storativity)[~np.isnan(resulting_storativity)], truncate_to_max_percentile)

            for elem in tqdm(self.mesh.elements, desc="Truncating porosity values to P1 and P99"):
                if math.isnan(upscaled_storativity[elem.local_id]):
                  upscaled_storativity[elem.local_id] = minimum_storativity
                elif upscaled_storativity[elem.local_id] > maximum_storativity:
                    upscaled_storativity[elem.local_id] = maximum_storativity
                elif upscaled_storativity[elem.local_id] <= minimum_storativity:
                    upscaled_storativity[elem.local_id] = minimum_storativity
                else:
                    continue

        vtk_storativity = np.asarray(self.mesh.elements)
        for local_id in upscaled_storativity:
            vtk_storativity[local_id] = upscaled_storativity[local_id]

        self.mesh.cell_data['upscaled_storativity'] = self.mesh.refactor_array_by_element_type(vtk_storativity)
        self.upscaled_storativity = upscaled_storativity

        return upscaled_storativity

    def export_fault_distances(self):
        """Exports the fault distances"""
        distance = {}

        for elem in tqdm(self.mesh.elements, desc="Computing distances"):
            distance[elem.local_id] = 0
            for fault in elem.associated_faults:
                fault_dict = elem.associated_faults[fault]
                distance[elem.local_id] += fault_dict['distance'] if 'distance' in fault_dict else 0

        vtk_distance = np.asarray(self.mesh.elements)
        for local_id in distance:
            vtk_distance[local_id] = distance[local_id]

        self.mesh.cell_data['distance'] =  self.mesh.refactor_array_by_element_type(vtk_distance)
        self.distance = distance

    def export_fracture_property(self, property='area'):
        property_dict = {}
        for elem in tqdm(self.mesh.elements, desc="Computing fracture properties"):
            property_dict[elem.local_id] = 0
            # print(elem.associated_fractures)
            for fracture in elem.associated_fractures:
                fracture_dict = elem.associated_fractures[fracture]
                property_dict[elem.local_id] += fracture_dict[property]

        vtk_property = np.asarray(self.mesh.elements)
        for local_id in property_dict:
            vtk_property[local_id] = property_dict[local_id]

        self.mesh.cell_data[property] = [vtk_property.tolist()]
        self.property_dict = property_dict

    def upscale_mesh_permeability(self,
                                  matrix_permeability=None,
                                  rho=1000,
                                  g=9.8,
                                  mu=8.9e-4,
                                  mode='full_tensor',
                                  truncate_to_min_percentile=5,
                                  truncate_to_max_percentile=95,
                                  truncate=True,
                                  ):

        matrix_permeability = {}

        for elem in tqdm(self.mesh.elements, desc="Creating permeability tensor for dummy anisotropic case"):
            matrix_permeability[elem.local_id] = np.ones([3, 3]) * 0.0

        matrix_permeability_tensor = matrix_permeability

        # Check correct size of matrix_permeability.
        # matrix_permeability_tensor = np.zeros(len(self.elements))
        #
        # if len(matrix_permeability) != len(self.elements):
        #     print("Incorrect size for matrix permeability. Size of variable doesn't match number of elements in the mesh.")
        #     break
        # else:
        #     for elem in tqdm(self.elements, desc="Check size of matrix permeability input"):
        #         if len(matrix_permeability[elem]) == 3:
        #             if np.shape(matrix_permeability[elem]) == (3,3):
        #                 print("Matrix Permeability Tensor (3,3) for Anisotropic case.")
        #                 continue
        #             else:
        #                 print("Matrix Permeability Tensor must be an np.array([3,3]) for Anisotropic case.")
        #         elif len(matrix_permeability[elem]) == 1:
        #             print("Matrix Permeability for Isotropic case.")
        #             matrix_permeability_tensor[elem] = np.zeros([3,3])
        #             matrix_permeability_tensor[elem][0,0] = matrix_permeability[elem]
        #             continue
        #         else:
        #             print("Incorrect Matrix Permeability Tensor. Must be an np.array([3,3]) for use in Anisotropic case or a single float/int for use in Isotropic case.")
        #             continue

        # UPSCALED PERMEABILITY
        fracture_hk = {}
        fault_hk = {}
        upscaled_hk = {}

        # For each fracture, compute permeability tensor,
        # and add it to the elements intersected by the fracture.
        for elem in tqdm(self.mesh.elements, desc="Upscaling fractures permeability"):
            element_volume = elem.volume
            element_porosity = elem.total_fracture_volume / elem.volume
            fracture_hk[elem.local_id] = np.zeros([3, 3])
            upscaled_hk[elem.local_id] = np.zeros([3, 3])
            fault_hk[elem.local_id] = np.zeros([3, 3])

            for frac_name in elem.associated_fractures:
                frac_dict = elem.associated_fractures[frac_name]
                frac = frac_dict['fracture']
                # n1 = math.cos(frac.dip * (math.pi / 180)) * math.sin(frac.dip_dir * (math.pi / 180))
                # n2 = math.cos(frac.dip * (math.pi / 180)) * math.cos(frac.dip_dir * (math.pi / 180))
                # n3 = -1 * math.sin(frac.dip * (math.pi / 180))
                n1 = self.dfn[frac].unit_normal_vector[0]
                n2 = self.dfn[frac].unit_normal_vector[1]
                n3 = self.dfn[frac].unit_normal_vector[2]
                #frac.hk = ((frac.aperture ** 2) * rho * g) / (12 * mu)
                self.dfn[frac].hk = self.dfn[frac].transmissivity / self.dfn[frac].aperture
                #self.dfn[frac].hk = (5.932E-8 * (np.log10(self.dfn[frac].size / 2.0)) ** 2) / self.dfn[frac].aperture

                if mode == 'isotropy':
                    # Add fracture permeability, weighted by the area that the fracture occupies in the element.
                    fracture_hk[elem.local_id][0, 0] += self.dfn[frac].hk * (frac_dict['volume'] / element_volume)  # Kxx

                else:  # 'anisotropy' in 'mode':
                    perm_tensor = np.zeros([3, 3])
                    # for i in range(1, 4):
                    #    for j in range(1, 4):
                    # Compute tensor
                    perm_tensor[0, 0] = self.dfn[frac].hk * ((n2 ** 2) + (n3 ** 2))
                    perm_tensor[0, 1] = self.dfn[frac].hk * (-1) * n1 * n2
                    perm_tensor[0, 2] = self.dfn[frac].hk * (-1) * n1 * n3
                    perm_tensor[1, 1] = self.dfn[frac].hk * ((n3 ** 2) + (n1 ** 2))
                    perm_tensor[1, 2] = self.dfn[frac].hk * (-1) * n2 * n3
                    perm_tensor[2, 2] = self.dfn[frac].hk * ((n1 ** 2) + (n2 ** 2))

                    if 'mode' == 'anisotropy_principals':
                        eigen_perm_tensor = np.diag(np.linalg.eig(perm_tensor)[0])
                        perm_tensor = eigen_perm_tensor

                    frac_volume_in_element = frac_dict['volume'] / element_volume
                    # Add fracture permeability, weighted by the area that the fracture occupies in the element.
                    fracture_hk[elem.local_id][0, 0] += (perm_tensor[0, 0] * frac_volume_in_element)
                    fracture_hk[elem.local_id][0, 1] += (perm_tensor[0, 1] * frac_volume_in_element)
                    fracture_hk[elem.local_id][0, 2] += (perm_tensor[0, 1] * frac_volume_in_element)
                    fracture_hk[elem.local_id][1, 0] += (perm_tensor[0, 1] * frac_volume_in_element)
                    fracture_hk[elem.local_id][1, 1] += (perm_tensor[1, 1] * frac_volume_in_element)
                    fracture_hk[elem.local_id][1, 2] += (perm_tensor[1, 2] * frac_volume_in_element)
                    fracture_hk[elem.local_id][2, 0] += (perm_tensor[0, 1] * frac_volume_in_element)
                    fracture_hk[elem.local_id][2, 1] += (perm_tensor[1, 2] * frac_volume_in_element)
                    fracture_hk[elem.local_id][2, 2] += (perm_tensor[2, 2] * frac_volume_in_element)

            # Sum permeability contribution from fractures and from matrix.

            upscaled_hk[elem.local_id] = fracture_hk[elem.local_id] + matrix_permeability_tensor[elem.local_id] * (
                    1 - element_porosity)

            if len(elem.associated_faults) > 0:
                for fault_name in elem.associated_faults:
                    fault = self.dfn.faults[fault_name]
                    # n1 = math.cos(frac.dip * (math.pi / 180)) * math.sin(frac.dip_dir * (math.pi / 180))
                    # n2 = math.cos(frac.dip * (math.pi / 180)) * math.cos(frac.dip_dir * (math.pi / 180))
                    # n3 = -1 * math.sin(frac.dip * (math.pi / 180))
                    # n1 = fault.unit_normal_vector[0]
                    # n2 = fault.unit_normal_vector[1]
                    # n3 = fault.unit_normal_vector[2]
                    # frac.hk = ((frac.aperture ** 2) * rho * g) / (12 * mu)
                    effective_aperture = fault.effective_aperture if fault.effective_aperture is not None else fault.aperture
                    fault.hk = fault.transmissivity / effective_aperture

                    if mode == 'isotropy':
                        # Add fault permeability.
                        fault_hk[elem.local_id][0, 0] += fault.hk

                #
                # else:  # 'anisotropy' in 'mode':
                #     perm_tensor = np.zeros([3, 3])
                #     # for i in range(1, 4):
                #     #    for j in range(1, 4):
                #     # Compute tensor
                #     perm_tensor[0, 0] = fault.hk * ((n2 ** 2) + (n3 ** 2))
                #     perm_tensor[0, 1] = fault.hk * (-1) * n1 * n2
                #     perm_tensor[0, 2] = fault.hk * (-1) * n1 * n3
                #     perm_tensor[1, 1] = fault.hk * ((n3 ** 2) + (n1 ** 2))
                #     perm_tensor[1, 2] = fault.hk * (-1) * n2 * n3
                #     perm_tensor[2, 2] = fault.hk * ((n1 ** 2) + (n2 ** 2))
                #
                #     if 'mode' == 'anisotropy_principals':
                #         eigen_perm_tensor = np.diag(np.linalg.eig(perm_tensor)[0])
                #         perm_tensor = eigen_perm_tensor
                #
                #     # Add fault permeability. Element porosity equals to 1 when intersected by faults.
                #     fault_hk[elem.local_id][0, 0] += perm_tensor[0, 0]
                #     fault_hk[elem.local_id][0, 1] += perm_tensor[0, 1]
                #     fault_hk[elem.local_id][0, 2] += perm_tensor[0, 1]
                #     fault_hk[elem.local_id][1, 0] += perm_tensor[0, 1]
                #     fault_hk[elem.local_id][1, 1] += perm_tensor[1, 1]
                #     fault_hk[elem.local_id][1, 2] += perm_tensor[1, 2]
                #     fault_hk[elem.local_id][2, 0] += perm_tensor[0, 1]
                #     fault_hk[elem.local_id][2, 1] += perm_tensor[1, 2]
                #     fault_hk[elem.local_id][2, 2] += perm_tensor[2, 2]

            # Sum permeability contribution from faults.
            upscaled_hk[elem.local_id] = upscaled_hk[elem.local_id] + fault_hk[elem.local_id]

        #Post-processing: Truncate values in each direction.
        if truncate:
            for i in range(0,3):
                for j in range(0,3):
                    resulting_hk = [upscaled_hk[local_id][0,0] for local_id in upscaled_hk]
                    minimum_hk = -1.0
                    min_percentile = truncate_to_min_percentile - 1
                    while minimum_hk <= 0:
                        minimum_hk = np.percentile(np.array(resulting_hk)[~np.isnan(resulting_hk)], min_percentile)
                        min_percentile = min_percentile + 1
                        print(min_percentile)

                    print("HK will be truncated to Percentile = " + str(min_percentile))

                    maximum_hk = np.percentile(np.array(resulting_hk)[~np.isnan(resulting_hk)], truncate_to_max_percentile)

                    for elem in tqdm(self.mesh.elements, desc="Truncating permeability values"):
                        if math.isnan(upscaled_hk[elem.local_id][i,j]):
                            upscaled_hk[elem.local_id][i,j] = minimum_hk
                        elif upscaled_hk[elem.local_id][i,j] > maximum_hk:
                            upscaled_hk[elem.local_id][i,j] = maximum_hk
                        elif upscaled_hk[elem.local_id][i,j] <= minimum_hk:
                            upscaled_hk[elem.local_id][i,j] = minimum_hk
                        else:
                            continue

        # Export values to VTK
        vtk_kxx = np.asarray(self.mesh.elements)
        vtk_kyy = np.asarray(self.mesh.elements)
        vtk_kzz = np.asarray(self.mesh.elements)
        vtk_kxy = np.asarray(self.mesh.elements)
        vtk_kxz = np.asarray(self.mesh.elements)
        vtk_kyz = np.asarray(self.mesh.elements)
        for local_id in upscaled_hk:
            vtk_kxx[local_id] = upscaled_hk[local_id][0, 0]
            vtk_kyy[local_id] = upscaled_hk[local_id][1, 1]
            vtk_kzz[local_id] = upscaled_hk[local_id][2, 2]
            vtk_kxy[local_id] = upscaled_hk[local_id][0, 1]
            vtk_kxz[local_id] = upscaled_hk[local_id][0, 2]
            vtk_kyz[local_id] = upscaled_hk[local_id][1, 2]

        self.mesh.cell_data['Kxx'] = self.mesh.refactor_array_by_element_type(vtk_kxx)
        self.mesh.cell_data['Kyy'] = self.mesh.refactor_array_by_element_type(vtk_kyy)
        self.mesh.cell_data['Kzz'] = self.mesh.refactor_array_by_element_type(vtk_kzz)
        self.mesh.cell_data['Kxy'] = self.mesh.refactor_array_by_element_type(vtk_kxy)
        self.mesh.cell_data['Kxz'] = self.mesh.refactor_array_by_element_type(vtk_kxz)
        self.mesh.cell_data['Kyz'] = self.mesh.refactor_array_by_element_type(vtk_kyz)

        self.upscaled_permeability = upscaled_hk

        return upscaled_hk

    def to_vtk(self, filename):
        """Exports the mesh and the upscaled variables to VTK"""
        self.mesh.to_vtk(filename)

    def porosity_to_csv(self, filename='./porosity.csv'):
        """Exports porosity values to csv

        Returns:

        """
        import csv
        logger.info(f"Exporting porosity to {filename}")
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for element_id in self.upscaled_porosity:
                centroid = self.mesh.elements[element_id].centroid
                porosity = self.upscaled_porosity[element_id]
                writer.writerow([centroid[0], centroid[1], centroid[2], porosity])

    def plot_porosity_histogram(self, filename='upscaled_porosity_histogram.png'):
        """Plots the upscaled porosity histogram."""
        logger.info(f'Plotting upscaled porosity histogram to {filename}')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.hist([self.upscaled_porosity[element_id] for element_id in self.upscaled_porosity], bins=100)
        return fig, ax

    def plot_hkx_histogram(self, filename='upscaled_hkx_histogram.png'):
        """Plots the upscaled hkx histogram."""
        logger.info(f'Plotting upscaled hkx histogram to {filename}')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.hist([self.upscaled_permeability[element_id][0, 0] for element_id in self.upscaled_permeability], bins=100)
        return fig, ax

    def plot_storativity_histogram(self, filename='upscaled_storativity_histogram.png'):
        """Plots the upscaled storativity histogram."""
        logger.info(f'Plotting upscaled storativity histogram to {filename}')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.hist([self.upscaled_storativity[element_id] for element_id in self.upscaled_storativity], bins=100)
        return fig, ax

    def export_intersection_stats(self, filename='intersection_stats.txt'):
        # Export the run_stats dictionary to file
        assert self.mesh.is_intersected, 'The mesh has not been intersected yet.'
        import json
        with open('run_stats.json', 'w') as fp:
            json.dump(self.mesh.find_intersection_stats, fp)

    def save(self, filename='upscaled_model.json'):
        """Save a copy of the class on a serialized pickle object"""
        logger.info(f'Saving a copy of the class to {filename}')
        # Create saving dictionary
        saving_dict = {}
        saving_dict['mesh'] = self.mesh.get_json()
        import jsonpickle
        with open(filename, 'w') as f:
            jsonpickle.encode(self, f)


    def to_json(self, filename='upscaler.json'):
        """Save a copy of the class on a serialized json object"""
        logger.info(f'Saving a copy of the class to {filename}')
        # Create saving dictionary
        saving_dict = {}
        saving_dict['mesh'] = self.mesh.get_json()
        saving_dict['dfn'] = self.dfn.get_json()
        intersected_points = []
        # for point_group in self.all_intersected_points:
        #     intersected_points.append([])
        #     for point in point_group:
        #         intersected_points[-1].append(point.get_json())
        # saving_dict['all_intersected_points'] = intersected_points
        import json
        with open(filename, 'w') as f:
            json.dump(saving_dict, f)

    @classmethod
    def from_json(cls, filename):
        """Load a serialized pickle object"""
        logger.info(f'Loading the upscaling class from {filename}')
        import json
        with open(filename, 'rb') as f:
            load_dict = json.load(f)
            mesh = MeshPreprocessor.from_dict(load_dict['mesh'])
            dfn = DfnPreprocessor.from_dict(load_dict['dfn'])
            loaded_class = cls(mesh=mesh, dfn=dfn, loading=True)
            # loaded_class.all_intersected_points = load_dict['all_intersected_points']
            return loaded_class

    def add_to_class(self, key, value, default=None):
        """Add an attribute to the class"""
        setattr(self, key, value)
        if value != default:
            logger.info(f'Added {key} = {value} to the class')





        


