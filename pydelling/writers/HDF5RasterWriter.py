import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .BaseWriter import BaseWriter


class HDF5RasterWriter(BaseWriter):
    def __init__(self,
                 filename,
                 dataset_name,
                 data=None,
                 times=0.0,
                 attributes={},
                 **kwargs,
                 ):
        super().__init__(self, data=data, **kwargs)
        if self.info["interpolation"]["type"] == "regular_mesh":
            if len(self.data.shape) == 1:
                self.data = self.transform_flatten_to_regular_mesh(self.data)
                self.data = np.array(self.data)
                plt.imshow(self.data[0, :, :])
                plt.show()
                self.data = np.swapaxes(np.array(self.data), 0, 1)
                self.data = np.swapaxes(self.data, 1, 2)
            elif len(self.data.shape) == 2:
                if self.data.shape[1] == 3:
                    self.data = self.centroid_transform_to_mesh()
                else:
                    self.data = self.transform_flatten_to_regular_mesh(data=self.data)
                    self.data = np.array(self.data)
                    plt.imshow(self.data[0, :, :])
                    plt.show()
                    self.data = np.swapaxes(np.array(self.data), 0, 1)
                    self.data = np.swapaxes(self.data, 1, 2)
            elif len(self.data.shape) == 3:
                _data = []
                for layer in self.data:
                    _data.append(np.array(self._centroid_transform_to_mesh(layer)))
                self.data = np.array(_data)
                self.data = np.swapaxes(np.array(_data), 0, 1)
                self.data = np.swapaxes(self.data, 1, 2)
        if self.data is not None:
            self.data_loaded = True
        self.region_name = dataset_name
        self.times = times
        self.attributes = attributes

    def transform_flatten_to_regular_mesh(self, data):
        aux_array = []
        if len(data.shape) == 2:
            for case in data:
                aux_array.append(np.reshape(case, (self.info["interpolation"]["n_y"], self.info["interpolation"]["n_x"])).T)
        elif len(data.shape) == 1:
            aux_array.append(np.reshape(data, (self.info["interpolation"]["n_y"], self.info["interpolation"]["n_x"])).T)
        return np.array(aux_array)

    def centroid_transform_to_mesh(self):
        assert len(self.data.shape) >= 2 and self.data.shape[1] == 3
        _data = self.data[:, 2]
        _data = np.reshape(_data, (self.info["interpolation"]["n_x"], self.info["interpolation"]["n_y"]))
        return _data

    def _centroid_transform_to_mesh(self, data):
        assert len(data.shape) >= 1 and data.shape[1] == 3
        _data = data[:, 2]
        _data = np.reshape(_data, (self.info["interpolation"]["n_x"], self.info["interpolation"]["n_y"]))
        return _data

    def add_default_attributes(self, hdf5_group: h5py.Dataset):
        dilatation_factor = self.info['interpolation']['dilatation_factor']
        l_x = np.abs(self.info["interpolation"]["x_max"] - self.info["interpolation"]["x_min"])
        l_x_dilatated = np.abs(self.info["interpolation"]["x_max"] - self.info["interpolation"]["x_min"]) * dilatation_factor
        delta_x = (l_x_dilatated - l_x) / 2.0
        dx = self.info["interpolation"]["d_x"] * dilatation_factor
        l_y = np.abs(self.info["interpolation"]["y_max"] - self.info["interpolation"]["y_min"])
        l_y_dilatated = np.abs(self.info["interpolation"]["y_max"] - self.info["interpolation"]["y_min"]) * dilatation_factor
        delta_y = (l_y_dilatated - l_y) / 2.0
        dy = self.info["interpolation"]["d_y"] * dilatation_factor

        hdf5_group.attrs.create('Dimension', self.attributes['Dimension'] if 'Dimension' in self.attributes else 'XY', dtype="S3")
        hdf5_group.attrs["Discretization"] = [dx, dy]
        hdf5_group.attrs["Origin"] = [self.info["interpolation"]["x_min"] - delta_x, self.info["interpolation"]["y_min"] - delta_y]
        hdf5_group.attrs["Interpolation_Method"] = "STEP"

    def run(self, filename=None):
        if filename is not None:
            self.filename = filename

        if self.check_data():
            if not os.path.exists(self.filename):
                h5temp = h5py.File(self.filename, "w")
                h5temp.close()
            else:
                # Delete the file if it exists
                os.remove(self.filename)
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
                        if attribute == "Dimension":
                            temp_group.attrs.create(attribute, self.attributes[attribute], dtype="S3")
                        else:
                            temp_group.attrs[attribute] = self.attributes[attribute]

        else:
            print("Couldn't find data to dump!")

    def add_dimension_attribute(self, dimension):
        self.attributes["Dimension"] = dimension
