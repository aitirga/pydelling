"""
Centroid file reader
"""
import numpy as np
from ..utils import globals

class CentroidReader:
    def __init__(self, filename,
                  var_pos=3,
                  var_name="var",
                  var_type=np.float32,
                  centroid_pos=(0, 3),
                  header=False):
        self.var_pos = var_pos
        self.var_name = var_name
        self.var_type = var_type
        self.centroid_pos = centroid_pos
        self.filename = filename
        self.header = header
        self.info = {}
        self.data = {}
        with open(filename) as opened_file:
            if self.header:
                opened_file.readline()  # For now, skips the header if it has
            self.read_file(opened_file, var_pos, var_name, var_type, centroid_pos)
        self.build_info()

    def read_file(self, opened_file,
                  var_pos=3,
                  var_name="var",
                  var_type=np.float32,
                  centroid_pos=(0, 3)):
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
            temp_centroid.append(data_row[centroid_pos[0]:centroid_pos[1]])
            temp_id.append([data_row[var_pos]])
        self.data["centroids"] = np.array(temp_centroid, dtype=np.float32)
        self.data[var_name] = np.array(temp_id, dtype=var_type)

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
        return np.hstack((self.data["centroids"], self.data[self.var_name]))

    def build_info(self):
        """
        Generates a dictionary containing the basic info of the read data
        :return:
        """
        self.info = {"n_cells": self.data["centroids"].shape[0],
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

