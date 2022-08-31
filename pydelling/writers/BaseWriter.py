import os

import numpy as np


class BaseWriter:
    info: dict
    def __init__(self, filename=None, var_name=None, data=None, region_name=None, **kwargs):
        self.data_loaded = False
        self.a_min = None
        self.a_max = None
        self.__dict__.update(**kwargs)
        self.filename = filename if filename else "test.dat"
        if var_name:
            self.var_name = var_name
        if region_name:
            self.region_name = region_name
        if data is not None:
            self.data = data
            self.data_loaded = True

    def set_data_limits(self, a_min=None, a_max=None):
        if a_min == 'None':
            a_min = None
        if a_min is not None:
            self.a_min = float(a_min)
        if a_max == 'None':
            a_max = None
        if a_max is not None:
            self.a_max = float(a_max)

    def apply_data_limits(self):
        print(f"Applying data limits to {self.data} with minimum value: {self.a_min} and maximum value: {self.a_max}")
        if not (self.a_min is None and self.a_max is None):
            try:
                self.data = np.clip(self.data, a_min=self.a_min, a_max=self.a_max)
                print(f"Resulting array: {self.data} Min value: {np.min(self.data)} Max value: {np.max(self.data)}")
            except Exception as e:
                print(f"INFO: Cannot apply minimum and maximum values")
                print(f"ERROR: {e}")

    def load_data(self, var_name=None, data=None, apply_data_limits=True):
        """
        Load data to export
        :return:
        """
        self.wipe_data()
        if var_name is not None and data is not None:
            self.var_name = var_name
            self.data = data
            self.data_loaded = True
        if self.has_data_loaded() and apply_data_limits:
            self.apply_data_limits()
        return self.data_loaded

    def wipe_data(self):
        self.var_name = None
        self.data_loaded = False
        self.data = None

    def check_data(self):
        return self.data_loaded

    def run(self, filename=None):
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

    def has_data_loaded(self):
        return self.data_loaded

