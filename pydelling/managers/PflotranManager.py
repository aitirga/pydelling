from .BaseManager import BaseManager


class PflotranManager(BaseManager):
    """This class extends the BaseManager class to manage PFLOTRAN related simulations.
    """
    def __init__(self, input_file: str):
        """This method initializes the class.
        """
        super().__init__(input_file)
        self.regions = self.get_regions()
        print(self.regions)

    def get_regions(self):
        """This method returns the regions of the simulation.
        """
        region_lines = self._find_tags('region')
        regions = []
        for line_idx in region_lines:
            line = self._get_line(line_idx)
            if self._get_parent_tag_name_(line_idx).lower() == 'region':
                regions.append(line.split()[1])
        return regions

    def _get_parent_tag_name_(self, line_index: int):
        """This method returns the parent tag of the line.
        """
        # Find the previous END tag
        has_end_tag = False
        temp_list = []
        temp_list.append(self._get_line(line_index))
        while not has_end_tag:
            line_index -= 1
            line = self._get_line(line_index)
            if len(line.split()) == 0:
                continue
            if line[0] == '#':
                continue
            temp_list.append(line)
            if 'end' in line.lower():
                has_end_tag = True
        return temp_list[-2].split()[0]



