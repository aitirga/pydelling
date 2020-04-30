"""
Centroid file reader
"""
import numpy as np
from ..utils import globals
from .BaseReader import BaseReader


class CentroidReader(BaseReader):
    def __init__(self, filename, var_pos=3, var_name="var", var_type=np.float32, centroid_pos=(0, 3), header=False):
        self.var_pos = None
        self.var = None
        self.var_name = None
        self.var_type = None
        self.centroid_pos = None
        self.header = None
        super().__init__(filename, var_pos=var_pos,
                         var_name=var_name,
                         var_type=var_type,
                         centroid_pos=centroid_pos,
                         header=header)

    def read_file(self, opened_file):
        """
        Reads the data and stores it inside the class
        :return:
        """
        if globals.config.general.verbose:
            print(f"Reading centroid file from {self.filename}")
        temp_centroid = []
        temp_id = []
        for line in opened_file.readlines():
            data_row = line.split()
            temp_centroid.append(data_row[self.centroid_pos[0]:self.centroid_pos[1]])
            temp_id.append([data_row[self.var_pos]])
        self.data = np.array(temp_centroid, dtype=np.float32)
        self.var = np.array(temp_id, dtype=self.var_type)

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
        return np.hstack((self.data, self.var))

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

