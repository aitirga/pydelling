"""
Interpolates a given set of points into a PFLOTRAN mesh
"""
import numpy as np
from scipy.interpolate import griddata
from .BaseInterpolator import BaseInterpolator
import logging
from PyFLOTRAN.utils.decorators import set_run



logger = logging.getLogger(__name__)


class SparseDataInterpolator(BaseInterpolator):
    @set_run
    def interpolate(self, **kwargs):
        logger.info(f"Interpolating data based on {self.info}")
        self.interpolated_data = griddata(self.data[:, 0:-1], self.data[:, -1], self.mesh, **kwargs)
        return self.get_data()

    def get_data(self):
        temp_array = np.reshape(self.interpolated_data, (self.interpolated_data.shape[0], 1))
        return np.concatenate((self.mesh, temp_array), axis=1)

    def change_min_value(self, min_value=None):
        logger.info(f"Equaling values <{min_value} to {min_value}")
        self.interpolated_data[self.interpolated_data < min_value] = min_value
        return self.interpolated_data
