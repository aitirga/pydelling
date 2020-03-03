"""
Base interface for a reader class
"""
import numpy as np
from ..utils import globals


class BaseReader:
    def __init__(self, filename):
        self.filename = filename
        self.info = {}
        self.data = None
        with open(filename) as opened_file:
            self.read_file(opened_file)

    def read_file(self, opened_file):
        """
        Reads the data and stores it inside the class
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

    def dump_to_csv(self, output_file):
        """
        Writes the data into a csv file
        :param output_file:
        :return:
        """
        print(f"Starting dump into {output_file}")
        f = open(output_file, "w")
        for data in self.data:
            f.write(f"{data[0]},{data[1]},{data[2]}\n")
        f.close()
        print(f"The data has been properly exported to the {output_file} file")

    def dump_to_wsv(self, output_file):
        print(f"Starting dump into {output_file}")
        f = open(output_file, "w")
        for data in self.data:
            f.write(f"{data[0]} {data[1]} {data[2]}\n")
        f.close()
        print(f"The data has been properly exported to the {output_file} file")
