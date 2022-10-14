from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydelling.readers.iGPReader import iGPReader
import logging

logger = logging.getLogger(__name__)


class RegionOperations:
    boundaries: dict
    region_dict: dict
    # Add iGPReader class methods to this class namespace

    def divide_topography_by_z(self: iGPReader, z: float,
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



