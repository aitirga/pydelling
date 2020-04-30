"""
Interpolates a given set of points into a PFLOTRAN mesh
"""
import numpy as np
from scipy.interpolate import griddata
from .BaseInterpolator import BaseInterpolator


class SparseDataInterpolator(BaseInterpolator):
    def interpolate(self, **kwargs):
        self.interpolated_data = griddata(self.data[:, 0:-1], self.data[:, -1], self.mesh, **kwargs)
        return self.get_data()

    def get_data(self):
        temp_array = np.reshape(self.interpolated_data, (self.interpolated_data.shape[0], 1))
        return np.concatenate((self.mesh, temp_array), axis=1)

