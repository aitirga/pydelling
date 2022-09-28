"""
Base interface for a reader class
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
import pydelling.utils.SubFishModule as subfish


class BaseReader:
    data: np.ndarray  # Hint of self.data array
    info: dict

    def __init__(self, filename=None,
                 header=False,
                 read_data=True,
                 **kwargs,
                 ):
        self.filename = Path(filename)
        self.info = {"reader": {}}
        self.data = None
        self.header = header
        self.__dict__.update(kwargs)
        if read_data:
            self.open_file(filename, **kwargs)

    def read_file(self, opened_file):
        """
        Reads the data and stores it inside the class
        :return:
        """
        pass

    def open_file(self, filename, **kwargs):
        with open(filename) as opened_file:
            if self.header:
                opened_file.readline()  # For now, skips the header if it has
            self.read_file(opened_file)
        self.build_info()

    def read_header(self, opened_file):
        """
        Reads the header of the file
        :return:
        """
        pass

    def get_data(self) -> np.ndarray:
        """
        Outputs the read data
        :return:
        """
        return np.array(0)

    def build_info(self):
        """
        Generates a dictionary containing the basic info of the read data
        :return:
        """
        self.info = {}

    def global_coords_to_local(self, x_local_to_global, y_local_to_global):
        """Converts global data coordinates into local"""
        assert len(self.data.shape) >= 2 and self.data.shape[1] >= 2, "Error in data shape"
        self.data[:, 0] -= x_local_to_global
        self.data[:, 1] -= y_local_to_global

    def local_coords_to_global(self, x_local_to_global, y_local_to_global):
        """Converts local data coordinates into global"""
        assert len(self.data.shape) >= 2 and self.data.shape[1] >= 2, "Error in data shape"
        self.data[:, 0] += x_local_to_global
        self.data[:, 1] += y_local_to_global

    def dump_to_csv(self, output_file, delimiter=","):
        """
        Writes the data into a csv file
        :param output_file:
        :return:
        """
        print(f"Starting dump into {output_file}")
        np.savetxt(output_file, self.get_data(), delimiter=delimiter)
        print(f"The data has been properly exported to the {output_file} file")

    def create_postprocess_dict(self):
        self.postprocessing_dict = Path().cwd() / "postprocess"
        self.postprocessing_dict.mkdir(exist_ok=True)

    def generate_subfish_data(self, subfish_dict: dict, unit_factor=1/(365 * 24), unit_name='d') -> pd.DataFrame:
        """
        This method reads a given subfish dict and returns the calculated results
        Args:
            subfish_dict: dictionary containing subfish module parameters
            unit_factor: factor multiplyin the results (originally computed in seconds)
        Returns:
            Array containing the times and computed values
        """
        if not subfish_dict:
            raise AttributeError('Please, provide a suitable subfish dict object')
        tang_sol = subfish.calculate_tang(subfish_dict)
        tang_sol[0] *= unit_factor
        tang_sol_pd = pd.DataFrame({f'Time [{unit_name}]': tang_sol[0],
                                    'Result [M]': tang_sol[1]
                                    }
                                   )
        return tang_sol_pd

    @property
    def values(self):
        return self.get_data()
