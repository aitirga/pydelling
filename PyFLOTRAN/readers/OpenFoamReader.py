"""
Base interface for a reader class
"""
import numpy as np
import logging
from PyFLOTRAN.readers import BaseReader

logger = logging.getLogger(__name__)


class OpenFoamReader(BaseReader):
    def read_file(self, opened_file):
        """
        Reads the data and stores it inside the class
        :return:
        """
        pass

    def open_file(self, filename):
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
