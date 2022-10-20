from .BaseManager import BaseManager
import logging
from typing import Union, List, Dict

logger = logging.getLogger(__name__)


class PflotranManager(BaseManager):
    """This class extends the BaseManager class to manage PFLOTRAN related simulations.
    """
    def __init__(self, input_file: str):
        """This method initializes the class.
        """
        super().__init__(input_file)
        self.regions_to_idx = {}

    def get_regions(self):
        """This method returns the regions of the simulation.
        """
        region_lines = self._find_tags('region')
        regions = []
        for line_idx in region_lines:
            line = self._get_line(line_idx)
            if self._get_parent_tag_name_(line_idx).lower() == 'region':
                regions.append(line.split()[1])
                self.regions_to_idx[line.split()[1]] = line_idx
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

    def get_region_file(self, region: str) -> Union[str, None]:
        """This method returns the file of the region.
        """
        self.get_regions()
        region_line = self.regions_to_idx[region]
        region_block = self._get_block_lines(region_line)
        for line in region_block:
            if 'file' in line.lower():
                return line.split()[1]
        return None

    def replace_region_file(self, region: str, new_file: str):
        """This method replaces the file of the region.
        """
        self.get_regions()
        old_file = self.get_region_file(region)
        region_line = self.regions_to_idx[region]
        region_block = self._get_block_lines(region_line)
        region_block_idx = self._get_block_line_idx(region_line)
        logger.info(f"Replacing region {region} file from '{old_file}' to '{new_file}'")
        for line_idx, line in zip(region_block_idx, region_block):
            if 'file' in line.lower():
                self._replace_line(line_index=line_idx, new_line=['FILE', new_file])




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

    def _get_block_lines(self, line_index: int):
        """This method returns the lines of the block.
        """
        # Find the previous END tag
        has_end_tag = False
        temp_list = []
        temp_list.append(self._get_line(line_index))
        while not has_end_tag:
            line_index += 1
            line = self._get_line(line_index)
            if len(line.split()) == 0:
                continue
            if line[0] == '#':
                continue
            temp_list.append(line)
            if 'end' in line.lower():
                has_end_tag = True
        return temp_list

    def _get_block_line_idx(self, line_index: int):
        """This method returns the lines of the block.
        """
        # Find the previous END tag
        has_end_tag = False
        temp_list = []
        temp_list.append(line_index)
        while not has_end_tag:
            line_index += 1
            line = self._get_line(line_index)
            if len(line.split()) == 0:
                continue
            if line[0] == '#':
                continue
            temp_list.append(line_index)
            if 'end' in line.lower():
                has_end_tag = True
        return temp_list

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

    @property
    def idx_to_regions(self):
        """This method returns the regions of the simulation.
        """
        return {v: k for k, v in self.regions_to_idx.items()}



