"""
This class implements a basic interface for Interpolation classes
"""
import numpy as np
import h5py
import os
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

    def dump_to_csv(self, filename=None, **kwargs):
        temp_array = np.reshape(self.interpolated_data, (self.interpolated_data.shape[0], 1))
        temp_array = np.concatenate((self.mesh, temp_array), axis=1)
        np.savetxt(filename, temp_array, **kwargs)
        if globals.config.general.verbose:
            print(f"Data has been dumped into {filename}")

        # return np.concatenate((self.mesh, temp_array), axis=1)

    def wipe_data(self):
        """Whipes data structure
        """
        self.data = []
        self.mesh = []
        self.interpolated_data = []



