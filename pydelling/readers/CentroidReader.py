"""
Centroid file reader
"""
import logging

import numpy as np
import pandas as pd

from .BaseReader import BaseReader

logger = logging.getLogger(__name__)


class CentroidReader(BaseReader):
    """This class reads a data file described by a set of centroids"""
    def __init__(self, filename, var_pos=3, var_name="var", var_type=np.float32, centroid_pos=(0, 3), header=False, separator=None):
        self.var_pos = var_pos
        self.var = None
        self.var_name = var_name
        self.var_type = var_type
        self.centroid_pos = centroid_pos
        self.header = header
        self.split_key = separator
        if self.var_pos:
            logger.info(f"Reading data using centroids in positions [{self.centroid_pos[0]}, {self.centroid_pos[1] - 1}] and data in position {self.var_pos}")
        else:
            logger.info(f"Reading data using centroids in positions [{self.centroid_pos[0]}, {self.centroid_pos[1] - 1}]")

        super().__init__(filename, var_pos=var_pos,
                         var_name=var_name,
                         var_type=var_type,
                         centroid_pos=centroid_pos,
                         header=header)

    def read_file(self, opened_file):
        """
        Reads the data and stores it inside the class
        """
        logger.info(f"Reading centroid file from {self.filename}")
        temp_centroid = []
        temp_id = []
        for line in opened_file.readlines():
            if self.split_key:
                data_row = line.split(self.split_key)
            else:
                data_row = line.split()
            temp_centroid.append(data_row[self.centroid_pos[0]:self.centroid_pos[1] + 1])
            if self.var_pos:
                temp_id.append([data_row[self.var_pos]])
        self.data = np.array(temp_centroid, dtype=np.float32)
        if self.var_pos:
            self.var = np.array(temp_id, dtype=self.var_type)


    def read_header(self):
        """
        TODO: Add the header reader of the centroid file
        Reads the header of the file
        :return:
        """
        pass

    def get_data(self, as_dataframe=False) -> pd.DataFrame:
        """
        Outputs the data
        :return: np.ndarray object containing centroid information and variable output
        """
        if as_dataframe:
            if self.var_pos:
                return pd.DataFrame(self.data, columns=['x', 'y', 'z', f"{self.var_name}"])
            else:
                return pd.DataFrame(self.data, columns=['x', 'y', 'z'])
        else:
            return self.data


    def build_info(self):
        """
        Generates a dictionary containing the basic info of the read data
        :return:
        """
        self.info["reader"] = {"n_cells": self.data.shape[0],
                     "filename": self.filename,
                     "var_name": self.var_name,
                     "var_position": self.var_pos}

    def to_csv(self, output_file, delimiter=","):
        """
        Writes the data into a csv file
        :param output_file:
        :return:
        """
        logger.info(f"Starting dump into {output_file}")
        np.savetxt(output_file, self.get_data(), delimiter=delimiter)
        logger.info(f"The data has been properly exported to the {output_file} file")

    def shift(self, direction=np.array([0.0, 0.0, 0.0])):
        """Shifts the data coordinates on the given direction
        Args:
            direction: direction vector
        """
        if type(direction) is list:
            direction = np.array(direction)
        self.data[:, 0:3] += direction

