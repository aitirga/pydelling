import os
import h5py
import numpy as np


class BaseWriter:
    info: dict

    def __init__(self, filename="data.dat", var_name=None, data=None, region_name=None, **kwargs):
        self.data_loaded = False
        self.__dict__.update(**kwargs)
        if filename is not None:
            self.filename = filename
        if var_name is not None:
            self.var_name = var_name
        if region_name is not None:
            self.region_name = region_name
        if data is not None:
            self.data = data
            self.data_loaded = True

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

    def dump_file(self, filename=None):
        if filename is not None:
            self.filename = filename
        if self.check_data():
            if not os.path.exists(self.filename):
                temp_writer = open(self.filename, "w")
                temp_writer.close()
            with open(self.filename) as temp_writer:
                if type(self.data) == np.ndarray:
                    np.savetxt(self.filename, self.data)
            self.info["writer"] = {"filename": self.filename}
        else:
            print("Couldn't find data to dump!")

    def remove_output_file(self, filename=None):
        if filename is None:
            filename = self.filename
        else:
            self.filename = filename
        try:
            os.remove(filename)
        except FileNotFoundError as ef:
            print(f"Nothing to delete!")
        except PermissionError as perr:
            print(f"ERROR: Cannot delete file: {perr}")
            exit(1)
