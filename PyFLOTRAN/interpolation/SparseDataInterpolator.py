"""
Interpolates a given set of points into a PFLOTRAN mesh
"""
import os

import h5py
import numpy as np
from scipy.interpolate import griddata

from .BaseInterpolator import BaseInterpolator


class SparseDataInterpolator(BaseInterpolator):
    def interpolate(self, **kwargs):
        self.interpolated_data = griddata(self.data[:, 0:3], self.data[:, 3], self.mesh, **kwargs)
        return self.get_data()

    def get_data(self):
        temp_array = np.reshape(self.interpolated_data, (self.interpolated_data.shape[0], 1))
        return np.concatenate((self.mesh, temp_array), axis=1)

    def dump_to_hdf5(self, filename=None, var_name=None, data=None):
        """
        Dumps the data into HDF5 format
        :return:
        """
        assert hasattr(self, "id_data"), "Unstructured ID data needs to be provided"
        if not os.path.exists(filename):
            tempfile = h5py.File(filename, "w")
            tempfile.close()
        with h5py.File(filename, "r+") as tempfile:
            if not "Cell Ids" in list(tempfile):
                tempfile.create_dataset("Cell Ids", data=self.id_data, dtype=np.int32)
            print(self.interpolated_data)
            tempfile.create_dataset(var_name, data=self.interpolated_data)


