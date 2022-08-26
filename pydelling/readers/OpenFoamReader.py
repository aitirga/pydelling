"""
Base interface for a reader class
"""
import logging

import numpy as np

from pydelling.readers import BaseReader

logger = logging.getLogger(__name__)
import Ofpp
from pydelling.config import config
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OpenFoamReader(BaseReader):
    def __init__(self, filename=None):
        self.filename = Path(filename) if filename else Path(config.open_foam_reader.filename)
        logger.info(f"Reading OpenFOAM mesh file from {self.filename}")
        super().__init__(filename=self.filename)


    def open_file(self, filename):
        self.mesh = Ofpp.FoamMesh(self.filename)

    @property
    def cell_centers(self) -> np.array:
        if not config.globals.is_cell_centers_read:
            try:
                self.mesh.read_cell_centres(str(self.filename / "0/C"))
                logger.info(f"Reading cell center locations from {self.filename / '0/C'}")
                config.globals.is_cell_centers_read = True
            except:
                logger.info(f"Reading cell center locations from {self.filename / 'constant/C'}")
                self.mesh.read_cell_centres(str(self.filename / "constant/C"))
                config.globals.is_cell_centers_read = True

        return self.mesh.cell_centres

    @property
    def cell_volumes(self) -> np.array:
        if not config.globals.is_cell_volumes_read:
            try:
                self.mesh.read_cell_volumes(str(self.filename / "0/V"))
                logger.info(f"Reading cell volume locations from {self.filename / '0/V'}")
                config.globals.is_cell_volumes_read = True
            except:
                logger.info(f"Reading cell volume locations from {self.filename / 'constant/V'}")
                self.mesh.read_cell_volumes(str(self.filename / "constant/V"))
                config.globals.is_cell_volumes_read = True

        return self.mesh.cell_volumes

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

