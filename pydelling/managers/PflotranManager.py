from .BaseManager import BaseManager


class PflotranManager(BaseManager):
    """This class extends the BaseManager class to manage PFLOTRAN related simulations.
    """
    def __init__(self, input_file: str):
        """This method initializes the class.
        """
        super().__init__(input_file)
        self.regions = self.get_regions()

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

    def get_simulation_time(self, time_unit: str = 'y'):
        """This method returns the output time series of the simulation"""
        time_lines = self._get_line(self._find_tags('FINAL_TIME')[0])
        value = time_lines.split()[1]
        value_unit = time_lines.split()[2]
        return self.convert_time(value=value, initial_unit=value_unit, final_unit=time_unit)

    def replace_simulation_time(self, new_time: float, time_unit: str = 'y'):
        """This method replaces the simulation time of the simulation.
        """
        time_lines = self._find_tags('FINAL_TIME')
        new_time = self.convert_time(value=new_time, initial_unit=time_unit, final_unit=time_unit)
        self._replace_line(line_index=time_lines[0], new_line=['FINAL_TIME', str(new_time), time_unit])

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

    # Class properties
    @property
    def final_time(self):
        """This method returns the final time of the simulation.
        """
        return self.get_simulation_time()

    @property
    def final_time_unit(self):
        """This method returns the final time unit of the simulation.
        """
        return self._get_line(self._find_tags('FINAL_TIME')[0]).split()[2]



