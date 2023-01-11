from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydelling.readers.iGPReader import iGPReader
import logging

logger = logging.getLogger(__name__)
import numpy as np
from pydelling.utils.utility_subclasses import SemistructuredFinder


class RegionOperations:
    boundaries: dict
    region_dict: dict
    # Add iGPReader class methods to this class namespace

    def divide_topography_by_z(self: iGPReader,
                               z: float,
                               region_name: str,
                               name_below='below',
                               name_above='above',
                               ):
        """
        Divide the topography into two regions, one above and one below a given z coordinate.

        Args:
            z: The z coordinate to divide the topography by.
            region_name: The name of the region to divide.
            name_below: The name of the region below the z coordinate.
            name_above: The name of the region above the z coordinate.

        Returns:
            None
        """
        boundary_faces = self.get_boundary_faces(region_name)
        name_below = name_below if name_below != 'below' else f"{region_name}_below"
        name_above = name_above if name_above != 'above' else f"{region_name}_above"
        logger.info(f'Dividing topography by z={z} into regions {name_below} and {name_above}')

        new_boundaries = {name_below: [],
                          name_above: []}

        for face_idx, face in enumerate(boundary_faces):
            if face.centroid[2] < z:
                new_boundaries[name_below].append(face_idx)
            else:
                new_boundaries[name_above].append(face_idx)

        # Update the boundaries dict
        for name in new_boundaries:
            self.boundaries[name] = [boundary_faces[idx] for idx in new_boundaries[name]]
        self.boundaries.pop(region_name)

        # Update the region dict
        get_old_region = self.region_dict[region_name]
        # Divide the region elements based on the found idx
        self.region_dict[name_below] = {'elements': get_old_region['elements'][new_boundaries[name_below]],
                                        'centroid_id': get_old_region['centroid_id'][new_boundaries[name_below] ],
                                        'length': len(new_boundaries[name_below])}
        self.region_dict[name_above] = {'elements': get_old_region['elements'][new_boundaries[name_above]],
                                        'centroid_id': get_old_region['centroid_id'][new_boundaries[name_above]],
                                        'length': len(new_boundaries[name_above])}

        # Remove the old region
        self.region_dict.pop(region_name)

    def get_nodes_from_x_y(self: iGPReader,
                           x: float,
                           y: float,
                           materials: list = None,
                           top_region_name: str = None,
                           tol: float = 5.0,
                           eps: float = 2.5,
                           min_samples: int = 20,
                           ) -> list:
        if not hasattr(self, 'is_subset_generated'):
            self._setup_cluster_and_subset(eps=eps,
                                           min_samples=min_samples,
                                           top_region_name=top_region_name,
                                           materials=materials,
                                           )
            self.is_subset_generated = True

        # # Filter out only the nodes with the given x and y coordinates
        #
        # node_id = [node_id for node_id in self.node_id_subset if abs(self.nodes[node_id][0] - x) <= tol and abs(self.nodes[node_id][1] - y) <= tol]
        # sorted_ids = sorted(node_id, key=lambda x: self.nodes[x][2])
        # return sorted_ids

    def _setup_cluster_and_subset(self: iGPReader,
                                    eps: float,
                                    min_samples: int,
                                    top_region_name: str = None,
                                    materials: list = None,
                                    ):
        logger.info(f'Setting up cluster engine (DBSCAN with eps={eps} and min_samples={min_samples}) and subset of nodes for materials {materials}')
        self.is_subset_generated = False
        temp_node_id = []
        if materials is not None:
            for material in materials:
                element_nodes = [self.element_nodes[element_id] for element_id in self.get_material_elements(material)]
                element_nodes = np.unique(np.array(element_nodes).flatten())
                temp_node_id.append(element_nodes)
        else:
            element_nodes = np.unique(np.array(self.element_nodes).flatten())
            temp_node_id.append(element_nodes)
        temp_node_id = np.concatenate(temp_node_id)
        self.node_id_subset = np.unique(temp_node_id)
        self.cluster_engine = SemistructuredFinder(self.nodes[self.node_id_subset], eps=eps, min_samples=min_samples)
        logger.info(f'A total of {self.cluster_engine.n_clusters} clusters were found in the subset of nodes. Play with eps (eps={eps}) and min_samples (min_samples={min_samples}) to find more clusters.')
        try:
            top_region_elements = self.get_region_nodes(top_region_name)
            if self.cluster_engine.n_clusters != len(top_region_elements):
                logger.warning(f'The number of clusters found ({self.cluster_engine.n_clusters}) does not match the number of elements in the top region ({len(top_region_elements)}).')
        except:
            logger.warning(f'No top region was provided. The number of clusters found ({self.cluster_engine.n_clusters}) may not match the number of elements in the top region.')

