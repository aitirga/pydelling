from pydelling.preprocessing.mesh_preprocessor import MeshPreprocessor
from pydelling.preprocessing.dfn_preprocessor import DfnPreprocessor
from pydelling.preprocessing.dfn_preprocessor import Fracture
import pydelling.preprocessing.mesh_preprocessor.geometry as geometry
from pydelling.utils.geometry_utils import compute_polygon_area
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class DfnUpscaler:
    def __init__(self, dfn: DfnPreprocessor, mesh: MeshPreprocessor, parallel=False, save_intersections=False):
        self.dfn: DfnPreprocessor = dfn
        self.mesh: MeshPreprocessor = mesh
        logger.info('The DFN and mesh objects have been set properly')
        self.all_intersected_points = []
        self.save_intersections = save_intersections
        self._intersect_dfn_with_mesh(parallel=parallel)

    def _intersect_dfn_with_mesh(self, parallel=False):
        """Runs the DfnUpscaler"""
        logger.info('Upscaling the DFN to the mesh')

        if not parallel:
            for fracture in tqdm(self.dfn, desc='Intersecting fractures with mesh', total=len(self.dfn)):
                self.find_intersection_points_between_fracture_and_mesh(fracture)
        else:
            from joblib import Parallel, delayed
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            logger.info(f'Running using {num_cores} cores')
            Parallel(n_jobs=num_cores)(delayed(self.find_intersection_points_between_fracture_and_mesh)(fracture) for fracture in tqdm(self.dfn, desc='Intersecting fractures with mesh', total=len(self.dfn)))

        if self.save_intersections:
            import csv
            with open('intersections.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'z'])
                for intersection in self.all_intersected_points:
                    for point in intersection:
                        writer.writerow([point.x, point.y, point.z])



    def find_intersection_points_between_fracture_and_mesh(self, fracture: Fracture, export_stats=False):
        """Finds the intersection points between a fracture and the mesh"""

        intersection_points = []
        kd_tree_filtered_elements = self.mesh.get_closest_mesh_elements(fracture.centroid, distance=fracture.size)
        counter = 0
        elements_filtered = 0
        for element in kd_tree_filtered_elements:
            element: geometry.BaseElement
            counter += 1
            absolute_distance = np.abs(fracture.distance_to_point(element.centroid))
            characteristic_length = np.power(element.volume, 1/3)
            if absolute_distance > 2.0 * characteristic_length:
                elements_filtered += 1
                continue

            intersection_points = element.intersect_with_fracture(fracture)
            if self.save_intersections:
                self.all_intersected_points.append(intersection_points)

            intersection_area = compute_polygon_area(intersection_points)
            fracture.intersection_dictionary[element.local_id] = intersection_area
            element.associated_fractures[fracture.local_id] = {
                'area': intersection_area,
                'volume': intersection_area * fracture.aperture,
                'fracture': fracture,
            }
            n_intersections = len(intersection_points)
            if not n_intersections in self.mesh.find_intersection_stats['intersection_points'].keys():
                self.mesh.find_intersection_stats['intersection_points'][n_intersections] = 0
            self.mesh.find_intersection_stats['intersection_points'][n_intersections] += 1
            self.mesh.find_intersection_stats['total_intersections'] += 1

        self.mesh.is_intersected = True


        return intersection_points

    def _compute_fracture_volume_in_elements(self):
        # Compute volume of fractures in each element.
        #self.elements.total_fracture_volume = np.zeros([len(elements)])
        for elem in tqdm(self.mesh.elements, desc="Computing fracture volume fractions"):
            for fracture in elem.associated_fractures:
                fracture_dict = elem.associated_fractures[fracture]
                # Attribute of the element: portion of element occupied by fractures.
                elem.total_fracture_volume += fracture_dict['volume']

    def upscale_mesh_porosity(self, matrix_porosity=None):
        # Compute upscaled porosity for each element.
        self._compute_fracture_volume_in_elements()
        upscaled_porosity = {}
        for elem in tqdm(self.mesh.elements, desc="Upscaling porosity"):
            upscaled_porosity[elem.local_id] = (elem.total_fracture_volume / elem.volume)# + (
                       # matrix_porosity[elem] * (1 - (elem.total_fracture_volume / elem.volume)))
            # if elem.total_fracture_volume > 0:
            #     pass

        vtk_porosity = np.asarray(self.mesh.elements)
        for local_id in upscaled_porosity:
            vtk_porosity[local_id] = upscaled_porosity[local_id]

        self.mesh.cell_data['upscaled_porosity'] = [vtk_porosity.tolist()]
        self.upscaled_porosity = upscaled_porosity

        return upscaled_porosity

    def upscale_mesh_permeability(self, matrix_permeability=None, rho=1000, g=9.8, mu=8.9e-4,
                                  mode='full_tensor'):

        matrix_permeability = {}

        for elem in tqdm(self.mesh.elements, desc="Creating permeability tensor for dummy anisotropic case"):
            matrix_permeability[elem.local_id] = np.ones([3, 3])*0.0

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
        fracture_perm = {}
        upscaled_perm = {}

        # For each fracture, compute permeability tensor,
        # and add it to the elements intersected by the fracture.
        for elem in tqdm(self.mesh.elements, desc="Upscaling permeability"):
            fracture_perm[elem.local_id] = np.zeros([3, 3])
            upscaled_perm[elem.local_id] = np.zeros([3, 3])
            for frac_name in elem.associated_fractures:
                frac_dict = elem.associated_fractures[frac_name]
                frac = frac_dict['fracture']
                # n1 = math.cos(frac.dip * (math.pi / 180)) * math.sin(frac.dip_dir * (math.pi / 180))
                # n2 = math.cos(frac.dip * (math.pi / 180)) * math.cos(frac.dip_dir * (math.pi / 180))
                # n3 = -1 * math.sin(frac.dip * (math.pi / 180))
                n1 = frac.unit_normal_vector[0]
                n2 = frac.unit_normal_vector[1]
                n3 = frac.unit_normal_vector[2]
                frac.perm = ((frac.aperture ** 2) * rho * g) / (12 * mu)

                if 'mode' == 'isotropy':
                    # Add fracture permeability, weighted by the area that the fracture occupies in the element.
                    fracture_perm[elem.local_id][0, 0] += frac.perm * frac_dict['volume'] / elem.volume #Kxx

                else: #'anisotropy' in 'mode':
                    perm_tensor = np.zeros([3, 3])
                    #for i in range(1, 4):
                    #    for j in range(1, 4):
                            # Compute tensor
                    perm_tensor[0, 0] = frac.perm * ((n2 ** 2) + (n3 ** 2))
                    perm_tensor[0, 1] = frac.perm * (-1) * n1 * n2
                    perm_tensor[0, 2] = frac.perm * (-1) * n1 * n3
                    perm_tensor[1, 1] = frac.perm * ((n3 ** 2) + (n1 ** 2))
                    perm_tensor[1, 2] = frac.perm * (-1) * n2 * n3
                    perm_tensor[2, 2] = frac.perm * ((n1 ** 2) + (n2 ** 2))

                    if 'mode' == 'anisotropy_principals':
                        eigen_perm_tensor = np.diag(np.linalg.eig(perm_tensor)[0])
                        perm_tensor = eigen_perm_tensor

                    # Add fracture permeability, weighted by the area that the fracture occupies in the element.
                    fracture_perm[elem.local_id][0, 0] += (perm_tensor[0, 0] * frac_dict['volume'] / elem.volume)
                    fracture_perm[elem.local_id][0, 1] += (perm_tensor[0, 1] * frac_dict['volume'] / elem.volume)
                    fracture_perm[elem.local_id][0, 2] += (perm_tensor[0, 1] * frac_dict['volume'] / elem.volume)
                    fracture_perm[elem.local_id][1, 0] += (perm_tensor[0, 1] * frac_dict['volume'] / elem.volume)
                    fracture_perm[elem.local_id][1, 1] += (perm_tensor[1, 1] * frac_dict['volume'] / elem.volume)
                    fracture_perm[elem.local_id][1, 2] += (perm_tensor[1, 2] * frac_dict['volume'] / elem.volume)
                    fracture_perm[elem.local_id][2, 0] += (perm_tensor[0, 1] * frac_dict['volume'] / elem.volume)
                    fracture_perm[elem.local_id][2, 1] += (perm_tensor[1, 2] * frac_dict['volume'] / elem.volume)
                    fracture_perm[elem.local_id][2, 2] += (perm_tensor[2, 2] * frac_dict['volume'] / elem.volume)

            # Sum permeability contribution from fractures and from matrix.
            upscaled_perm[elem.local_id] = fracture_perm[elem.local_id] + matrix_permeability_tensor[elem.local_id] * (1 - (elem.total_fracture_volume / elem.volume))

        #Export values to VTK
        vtk_kxx = np.asarray(self.mesh.elements)
        vtk_kyy = np.asarray(self.mesh.elements)
        vtk_kzz = np.asarray(self.mesh.elements)
        vtk_kxy = np.asarray(self.mesh.elements)
        vtk_kxz = np.asarray(self.mesh.elements)
        vtk_kyz = np.asarray(self.mesh.elements)
        for local_id in upscaled_perm:
            vtk_kxx[local_id] = upscaled_perm[local_id][0, 0]
            vtk_kyy[local_id] = upscaled_perm[local_id][1, 1]
            vtk_kzz[local_id] = upscaled_perm[local_id][2, 2]
            vtk_kxy[local_id] = upscaled_perm[local_id][0, 1]
            vtk_kxz[local_id] = upscaled_perm[local_id][0, 2]
            vtk_kyz[local_id] = upscaled_perm[local_id][1, 2]

        self.mesh.cell_data['Kxx'] = [vtk_kxx.tolist()]
        self.mesh.cell_data['Kyy'] = [vtk_kyy.tolist()]
        self.mesh.cell_data['Kzz'] = [vtk_kzz.tolist()]
        self.mesh.cell_data['Kxy'] = [vtk_kxy.tolist()]
        self.mesh.cell_data['Kxz'] = [vtk_kxz.tolist()]
        self.mesh.cell_data['Kyz'] = [vtk_kyz.tolist()]

        self.upscaled_permeability = upscaled_perm

        return upscaled_perm

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

    def export_intersection_stats(self, filename='intersection_stats.txt'):
        # Export the run_stats dictionary to file
        assert self.mesh.is_intersected, 'The mesh has not been intersected yet.'
        import json
        with open('run_stats.json', 'w') as fp:
            json.dump(self.mesh.find_intersection_stats, fp)