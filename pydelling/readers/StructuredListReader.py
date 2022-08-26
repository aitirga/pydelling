"""
Centroid file reader
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pydelling.config import config
from .BaseReader import BaseReader

logger = logging.getLogger(__name__)


class StructuredListReader(BaseReader):
    data: pd.DataFrame
    def __init__(self, filename=None, var_pos=3, var_name="var", var_type=np.float32, centroid_pos=(0, 3), header=False):
        self.var_pos = None
        self.var = None
        self.var_name = None
        self.var_type = None
        self.centroid_pos = None
        self.header = None
        self.filename = Path(filename) if filename else Path(config.structured_list_reader.filename)
        self.data: pd.DataFrame
        super().__init__(self.filename, var_pos=var_pos,
                         var_name=var_name,
                         var_type=var_type,
                         centroid_pos=centroid_pos,
                         header=header)

    def open_file(self, filename):
        self.read_file()

    def read_file(self):
        logger.info(f"Reading list file from {self.filename.stem}")
        _temp_array = []
        with open(config.structured_list_reader.filename, "r") as reading_file:
            for _ in range(config.structured_list_reader.header_offset):
                _read = reading_file.readline()
            for nz in range(config.structured_list_reader.nz):
                for ny in range(config.structured_list_reader.ny):
                    for nx in range(config.structured_list_reader.nx):
                        read_value = reading_file.readline().split()[0]
                        p = np.array([config.structured_list_reader.ox + nx * config.structured_list_reader.dx,
                                      config.structured_list_reader.oy + ny * config.structured_list_reader.dy,
                                      config.structured_list_reader.oz + nz * config.structured_list_reader.dz,
                                      ])  # Position vector
                        _temp_array.append(np.append(p, read_value))
                        # idx = nx + config.structured_list_reader.nx * ny + config.structured_list_reader.nx * config.structured_list_reader.ny * nz + config.structured_list_reader.header_offset
                        # _temp_array.append(np.array())
        grain_array = pd.DataFrame(_temp_array, columns=["x", "y", "z", "v"])
        # grain_array[grain_array["v"] < config.structured_list_reader.min_value] = config.structured_list_reader.min_value
        self.data = grain_array

    def read_header(self):
        """
        TODO: Add the header reader of the centroid file
        Reads the header of the file
        :return:
        """
        pass


    def get_data(self) -> np.ndarray:
        """
        Outputs the data
        :return: np.ndarray object containing centroid information and variable output
        """
        return self.data.values.astype(np.float)

    @property
    def coordinates(self):
        return self.data[["x", "y", "x"]].values.astype(np.float)

    @property
    def values(self):
        return self.data[["v"]].values.astype(np.float)

    def build_info(self):
        """
        Generates a dictionary containing the basic info of the read data
        :return:
        """
        self.info["reader"] = {"n_cells": self.data.shape[0],
                     "filename": self.filename,
                     "var_name": self.var_name,
                     "var_position": self.var_pos}

    def dump_to_csv(self, output_file, delimiter=","):
        """
        Writes the data into a csv file
        :param output_file:
        :return:
        """
        print(f"Starting dump into {output_file}")
        np.savetxt(output_file, self.get_data(), delimiter=delimiter)
        print(f"The data has been properly exported to the {output_file} file")

