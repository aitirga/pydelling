import os
import h5py
import numpy as np
from .BaseWriter import BaseWriter


class HDF5RasterWriter(BaseWriter):
    def __init__(self, filename, data=None, region_name="region", times=0.0, attributes={}, **kwargs):
        super().__init__(self, data=data, **kwargs)
        if self.info["interpolation"]["type"] == "regular_mesh":
            self.centroid_transform_to_mesh()
        if self.data is not None:
            self.data_loaded = True
        self.region_name = region_name
        self.times = times
        self.attributes = attributes

    def centroid_transform_to_mesh(self):
        assert len(self.data.shape) >= 2 and self.data.shape[1] == 3
        _data = self.data[:, 2]
        self.data = np.reshape(_data, (self.info["interpolation"]["n_x"], self.info["interpolation"]["n_y"]))

    def add_default_attributes(self, hdf5_group: h5py.Dataset):
        hdf5_group.attrs.create('Dimension', "XY", dtype="S3")
        hdf5_group.attrs["Discretization"] = [self.info["interpolation"]["d_x"], self.info["interpolation"]["d_y"]]
        hdf5_group.attrs["Origin"] = [self.info["interpolation"]["x_min"], self.info["interpolation"]["y_min"]]
        hdf5_group.attrs["Interpolation_Method"] = "STEP"

    def dump_file(self,
                  filename=None,
                  remove_if_exists=False):
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
                temp_group = h5temp.create_group(name=self.region_name)
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
