import os
import h5py
import numpy as np
from .BaseWriter import BaseWriter


class HDF5RasterWriter(BaseWriter):
    def __init__(self, filename, data=None, region_name="region", times=0.0, attributes={}, **kwargs):
        super().__init__(self, data=data, **kwargs)
        if self.info["interpolation"]["type"] == "regular_mesh":
            if len(self.data.shape) == 2:
                self.data = self.centroid_transform_to_mesh()
            if len(self.data.shape) == 3:
                _data = []
                for layer in self.data:
                    _data.append(self._centroid_transform_to_mesh(layer))
                self.data = np.swapaxes(np.array(_data), 0, 1)
                self.data = np.swapaxes(self.data, 1, 2)
        if self.data is not None:
            self.data_loaded = True
        self.region_name = region_name
        self.times = times
        self.attributes = attributes

    def centroid_transform_to_mesh(self):
        print(self.data)
        assert len(self.data.shape) >= 2 and self.data.shape[1] == 3
        _data = self.data[:, 2]
        _data = np.reshape(_data, (self.info["interpolation"]["n_x"], self.info["interpolation"]["n_y"]))
        return _data

    def _centroid_transform_to_mesh(self, data):
        assert len(data.shape) >= 2 and data.shape[1] == 3
        _data = data[:, 2]
        _data = np.reshape(_data, (self.info["interpolation"]["n_x"], self.info["interpolation"]["n_y"]))
        return _data


    def add_default_attributes(self, hdf5_group: h5py.Dataset):
        hdf5_group.attrs.create('Dimension', "XY", dtype="S3")
        hdf5_group.attrs["Discretization"] = [self.info["interpolation"]["d_x"], self.info["interpolation"]["d_y"]]
        hdf5_group.attrs["Origin"] = [self.info["interpolation"]["x_min"], self.info["interpolation"]["y_min"]]
        hdf5_group.attrs["Interpolation_Method"] = "STEP"

    def dump_file(self, filename=None):
        if filename is not None:
            self.filename = filename

        if self.check_data():
            if not os.path.exists(self.filename):
                h5temp = h5py.File(self.filename, "w")
                h5temp.close()
            with h5py.File(self.filename, "r+") as h5temp:
                try:
                    temp_group = h5temp.create_group(name=self.region_name)
                except ValueError as e:
                    print(f"ERROR writing HDF5 file: {e}")
                    print(f"INFO: Possible solution: use remove_output_file(filename=\"\")")
                    exit(1)
                temp_group.create_dataset("Times", data=self.times)
                temp_group.create_dataset("Data", data=self.data)
                # Adds default attributes to the group
                self.add_default_attributes(temp_group)
                # Extends the default attributes to the ones defined by the user
                if self.attributes is not {}:
                    for attribute in self.attributes:
                        temp_group.attrs[attribute] = self.attributes[attribute]
        else:
            print("Couldn't find data to dump!")
