
class RegionOperations:
    boundaries: dict
    region_dict: dict

    def divide_topography_by_z(self, z: float,
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


