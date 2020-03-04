"""
This class implements a basic interface for Interpolation classes
"""
import numpy as np
from ..utils import globals

class BaseInterpolator:
    def __init__(self,
                 interpolation_data=None,
                 mesh_data=None):
        self.data = []
        self.mesh = []
        self.interpolated_data = []
        if interpolation_data is not None:
            self.add_data(data=interpolation_data)
        if mesh_data is not None:
            self.add_mesh(data=mesh_data)

    def add_data(self, data):
        """
        Add a dataset that will be used to interpolate
        :return:
        """
        if self.data == []:
            self.data = np.array(data)
        else:
            self.data = np.vstack((self.data, data))

    def add_mesh(self, data):
        """
        Add the set of points on which interpolation will be performed
        :return:
        """
        if data.shape[1] > 3:
            data = data[:, 0:3]
        if self.mesh == []:
            self.mesh = np.array(data)
        else:
            self.mesh = np.vstack((self.mesh, data))

    def interpolate(self):
        """
        Runs the interpolation algorithm.
        :return:
        """
        self.interpolated_data = self.mesh

    def get_data(self):
        """
        Returns interpolated data
        :return:
        """
        return self.interpolated_data

    def dump_to_hdf5(self):
        """
        Dumps the data into HDF5 format
        :return:
        """
        pass



