"""
Base interface for a reader class
"""
import numpy as np


class BaseReader:
    def __init__(self, filename):
        self.filename = filename
        self.info = {}
        self.data = {}
        with open(filename) as opened_file:
            self.read_file(opened_file)
        self.build_info()

    def read_file(self, opened_file):
        """
        Reads the data and stores it inside the class
        :return:
        """
        pass

    def read_header(self):
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

    def dump_to_csv(self, output_file, delimiter=","):
        """
        Writes the data into a csv file
        :param output_file:
        :return:
        """
        print(f"Starting dump into {output_file}")
        np.savetxt(output_file, self.get_data(), delimiter=delimiter)
        print(f"The data has been properly exported to the {output_file} file")
