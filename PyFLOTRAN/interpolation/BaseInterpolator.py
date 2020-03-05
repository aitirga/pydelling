"""
This class implements a basic interface for Interpolation classes
"""
import os

import h5py
import numpy as np


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

    def add_mesh(self, data, id_index=3):
        """
        Add the set of points on which interpolation will be performed
        :return:
        """
        if data.shape[1] > 3:
            temp_id_data = data[:, id_index]
            temp_data = data[:, 0:3]
        else:
            temp_data = data

        if self.mesh == []:
            self.mesh = np.array(temp_data)
            if data.shape[1] > 3:
                self.id_data = np.array(temp_id_data)
        else:
            self.mesh = np.vstack((self.mesh, temp_data))
            if data.shape[1] > 3:
                self.id_data = np.vstack((self.id_data, temp_id_data))

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

    def dump_to_hdf5(self, filename=None, var_name=None, data=None):
        """
        Dumps the data into HDF5 format
        :return:
        """
        if data == None:
            data = self.interpolated_data
        if not os.path.exists(filename):
            tempfile = h5py.File(filename, "w")
            tempfile.close()
        with h5py.File(filename, "r+") as tempfile:
            tempfile.create_dataset(var_name, data=data)

    def whipe_data(self):
        """Whipes data structure
        """
        self.data = []
        self.mesh = []
        self.interpolated_data = []



