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
    divide_over_direction = None
    @set_run
    def run(self, method='nearest', divide_over_direction=None, **kwargs):
        """
        Interpolates the data using the given method.
        Args:
            method: interpolation method
            divide_over_direction: Divides the data into smaller chunks if the data is too large.
            **kwargs:

        Returns: interpolated data
        """
        if not divide_over_direction:
            logger.info(f"Interpolating data based on {self.info}")
            self.interpolated_data = griddata(self.data[:, 0:-1], self.data[:, -1], self.mesh, method=method, **kwargs)
            return self.get_data()
        else:
            logger.info(f"Dividing data into smaller chunks")
            # Divide the data depending on the given direction
            # For now, only divide in the x direction using the mean value
            mesh_x_plus: np.ndarray = self.mesh[self.mesh[:, 0] >= self.mesh[:, 0].mean()]
            mesh_x_minus: np.ndarray = self.mesh[self.mesh[:, 0] < self.mesh[:, 0].mean()]
            data_x_plus: np.ndarray = self.data[self.data[:, 0] >= self.data[:, 0].mean()]
            data_x_minus: np.ndarray = self.data[self.data[:, 0] < self.data[:, 0].mean()]
            # Interpolate the data
            interpolate_plus = griddata(data_x_plus[:, 0:-1], data_x_plus[:, -1], mesh_x_plus, method=method, **kwargs)
            interpolate_minus = griddata(data_x_minus[:, 0:-1], data_x_minus[:, -1], mesh_x_minus, method=method, **kwargs)
            # Combine the data
            self.interpolated_data = np.concatenate((interpolate_plus, interpolate_minus), axis=0)
            return self.get_data()




    def get_data(self):
        temp_array = np.reshape(self.interpolated_data, (self.interpolated_data.shape[0], 1))
        return np.concatenate((self.mesh, temp_array), axis=1)

    def change_min_value(self, min_value=None):
        logger.info(f"Equaling values <{min_value} to {min_value}")
        self.interpolated_data[self.interpolated_data < min_value] = min_value
        return self.interpolated_data
