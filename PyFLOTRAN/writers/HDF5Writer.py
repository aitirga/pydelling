import os
import h5py


class HDF5Generator:
    def __init__(self, filename="h5_dump.hdf5", var_name=None, data=None):
        self.data_loaded = False
        if filename is not None:
            self.filename = filename
        if var_name is not None:
            self.var_name = var_name
        if data is not None:
            self.data = data

    def load_data(self, var_name=None, data=None):
        """
        Load data to export
        :return:
        """

        self.wipe_data()
        if var_name is not None and data is not None:
            self.var_name = var_name
            self.data = data
            self.data_loaded = True
        return self.data_loaded

    def wipe_data(self):
        self.var_name = None
        self.data_loaded = False
        self.data = None

    def check_data(self):
        return self.data_loaded

    def dump_file(self, filename=None, remove_if_exists=False):
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
        else:
            print("Couldn't find data to dump!")
