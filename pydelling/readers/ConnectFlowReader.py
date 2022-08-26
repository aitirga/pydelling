"""
Base interface for a reader class
"""
import logging

import numpy as np

from pydelling.readers import BaseReader

logger = logging.getLogger(__name__)
from pydelling.config import config
import logging
from pathlib import Path
import pandas as pd
logger = logging.getLogger(__name__)
from linecache import getline


class ConnectFlowReader(BaseReader):
    def __init__(self, filename=None):
        self.filename = Path(filename) if filename else Path(config.open_foam_reader.filename)
        logger.info(f"Reading ConnectFlow mesh file from {self.filename}")
        super().__init__(filename=self.filename)

    def open_file(self, filename):
        filename_string = str(filename)
        header = getline(filename_string, 1).split()
        self.n_nodes = int(header[1])
        self.n_elements = int(header[3])
        self.n_mat = int(header[5])
        with open(filename, "r") as opened_file:
            _nodes = [line.rstrip().split() for line in opened_file.readlines()[2: 2 + self.n_nodes - 1]]
            self._mesh_nodes = pd.DataFrame(np.array(_nodes).astype(np.float), columns=["index", "x", "y", "z"])
            self._mesh_nodes = self.mesh_nodes.set_index("index")
        self.build_info()

    @property
    def mesh_nodes(self) -> pd.DataFrame:
        assert hasattr(self, "_mesh_nodes"), "Mesh has not been properly assigned"
        return self._mesh_nodes


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
        min_x = self._mesh_nodes["x"].min()
        max_x = self._mesh_nodes["x"].max()
        min_y = self._mesh_nodes["y"].min()
        max_y = self._mesh_nodes["y"].max()
        min_z = self._mesh_nodes["z"].min()
        max_z = self._mesh_nodes["z"].max()
        self.info = {
            "bounds": {
                "x": [min_x, max_x],
                "y": [min_y, max_y],
                "z": [min_z, max_z],
            }
        }
        self.info.update({
            "span": {
                "x": self.info["bounds"]["x"][1] - self.info["bounds"]["x"][0],
                "y": self.info["bounds"]["y"][1] - self.info["bounds"]["y"][0],
                "z": self.info["bounds"]["z"][1] - self.info["bounds"]["z"][0],
            }
        })

    def get_bounds(self):
        """
        Returns the x, y and z axis bounds
        Returns:
            A list containing the bounds
        """
        return self.info["bounds"]


    def get_span(self):
        """
        Returns the x, y and z axis span
        Returns:
            A list containing the span
        """
        return self.info["span"]