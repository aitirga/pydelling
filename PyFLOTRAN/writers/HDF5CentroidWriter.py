import os
import h5py
from .BaseWriter import BaseWriter
import numpy as np

class HDF5CentroidWriter(BaseWriter):
    def run(self, filename=None, remove_if_exists=False):
        if filename is not None:
            self.filename = filename
        if remove_if_exists:
            try:
                os.remove(self.filename)
            except FileNotFoundError as ef:
                print("Nothing to overwrite!")

        if self.check_data():
            if not os.path.exists(self.filename):
                h5temp = h5py.File(self.filename, "w")
                h5temp.close()
            with h5py.File(self.filename, "r+") as h5temp:
                h5temp.create_dataset(self.var_name, data=self.data)
                cell_id = np.array([index + 1 for index in range(len(self.data))])
                h5temp.create_dataset("CellID", data=cell_id)
        else:
            print("Couldn't find data to dump!")
