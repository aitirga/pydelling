from .BaseStudy import BaseStudy
import logging
from typing import Union, List, Dict

logger = logging.getLogger(__name__)


class PflotranStudy(BaseStudy):
    """This class extends the BaseStudy class to manage PFLOTRAN related simulations.
    """
    def __init__(self, input_file: str, *args, **kwargs):
        """This method initializes the class.
        """
        super().__init__(input_file, *args, **kwargs)
        self.regions_to_idx = {}
        self.datasets_to_idx = {}

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

    def get_checkpoint(self):
        """This method returns the checkpoint times of the simulation.
        """
        checkpoint_lines = self._find_tags('CHECKPOINT')
        if len(checkpoint_lines) == 0:
            return None
        checkpoint_line = checkpoint_lines[0]
        checkpoint_block = self._get_block_lines(checkpoint_line)
        for line in checkpoint_block:
            if 'times' in line.lower():
                return line
        return None

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

    def get_datasets(self) -> List[str]:
        """This method returns the datasets of the simulation.
        """
        datasets = []
        for line_idx in self._find_tags('DATASET'):
            line = self._get_line(line_idx)
            if line.split()[0].lower() == 'hdf5_dataset_name':
                continue
            if self._get_parent_tag_name_(line_idx).lower() == 'dataset':
                datasets.append(line.split()[1])
                self.datasets_to_idx[line.split()[1]] = line_idx
        return datasets



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

    def add_checkpoint(self, times: Union[float, List[float]], time_unit: str = 'y'):
        """This method adds a checkpoint to the simulation.
        """
        logger.info(f"Adding checkpoint at {times} {time_unit}")
        if isinstance(times, float):
            times = [times]
        times = [self.convert_time(value=time, initial_unit=time_unit, final_unit=time_unit) for time in times]
        times = [str(time) for time in times]
        # Check if the checkpoint tag is already in the input file
        if self.has_tag('CHECKPOINT'):
            checkpoint_line = self._find_tags('CHECKPOINT')[0]
            checkpoint_block = self._get_block_lines(checkpoint_line)
            checkpoint_block_idx = self._get_block_line_idx(checkpoint_line)
            for line_idx, line in zip(checkpoint_block_idx, checkpoint_block):
                if 'times' in line.lower():
                    self._replace_line(line_index=line_idx, new_line=['TIMES', time_unit, *times])
        else:
            # Find simulation block
            simulation_block_idx = self._get_block_line_idx(self._find_tags('SIMULATION')[0])
            # Find the last line of the simulation block
            last_line_idx = simulation_block_idx[-1] - 1
            # Add the checkpoint block
            self._add_line(line_index=last_line_idx, new_line=['CHECKPOINT'])
            self._add_line(line_index=last_line_idx + 1, new_line=['TIMES', time_unit, *times])
            self._add_line(line_index=last_line_idx + 2, new_line=['FORMAT', 'HDF5'])
            self._add_line(line_index=last_line_idx + 3, new_line=['/'])

    def add_restart(self, filename: str):
        # Find simulation block
        simulation_block_idx = self._get_block_line_idx(self._find_tags('SIMULATION')[0])
        # Find the last line of the simulation block
        last_line_idx = simulation_block_idx[-1] - 1
        # Add the checkpoint block
        self._add_line(line_index=last_line_idx, new_line=['RESTART'])
        self._add_line(line_index=last_line_idx + 1, new_line=['FILENAME', filename])
        self._add_line(line_index=last_line_idx + 2, new_line=['/'])

    def add_dataset(self, name: str, filename: str, hdf5_dataset_name: str):
        """This method adds a dataset to the simulation.
        """
        logger.info(f"Adding dataset {name} to the simulation")
        # Find simulation block
        if name in self.get_datasets():
            idx = self.datasets_to_idx[name]
            inside_block = self._get_block_line_idx(idx)
            for line_idx in inside_block:
                line = self._get_line(line_idx)
                if 'file' in line.lower():
                    self._replace_line(line_index=line_idx, new_line=['FILENAME', filename])
                if 'hdf5_dataset' in line.lower():
                    self._replace_line(line_index=line_idx, new_line=['HDF5_DATASET_NAME', hdf5_dataset_name])
        else:
            logger.warning(f"Dataset {name} not found in the simulation. Adding it.")
            subsurface_block_start = self.get_subsurface_idx()
            # Find the last line of the simulation block
            last_line_idx = subsurface_block_start + 2
            # Add the checkpoint block
            self._add_line(line_index=last_line_idx, new_line=['# Automatically added by pydelling'])
            self._add_line(line_index=last_line_idx + 1, new_line=['DATASET', name])
            self._add_line(line_index=last_line_idx + 2, new_line=['FILENAME', filename])
            self._add_line(line_index=last_line_idx + 3, new_line=['HDF5_DATASET_NAME', hdf5_dataset_name])
            self._add_line(line_index=last_line_idx + 4, new_line=['END'])

    def get_subsurface_idx(self) -> int:
        """This method returns the index of the subsurface tag.
        """
        subsurface_idx = self._find_tags('SUBSURFACE')
        for idx in subsurface_idx:
            if self._get_parent_tag_name_(idx) == 'SUBSURFACE':
                return idx

    def has_tag(self, tag: str):
        """This method returns True if the tag is in the input file.
        """
        return len(self._find_tags(tag)) > 0

    def _get_parent_tag_name_(self, line_index: int):
        """This method returns the parent tag of the line.
        """
        # Find the previous END tag
        has_end_tag = False
        temp_list = []
        temp_list.append(self._get_line(line_index))
        while not has_end_tag:
            line = self._get_line(line_index)
            line_index -= 1
            if len(line.split()) == 0:
                continue
            if line[0] == '#':
                continue
            if 'subsurface' in line.lower():
                has_end_tag = True
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



