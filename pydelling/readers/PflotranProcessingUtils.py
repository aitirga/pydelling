import numpy as np
import pandas as pd
import h5py

class PflotranProcessingUtils:
    """This class contains utility functions to use on the PflotranReader class"""
    variables: list
    coordinates: np.ndarray
    data: h5py.File
    # data:
    axis_translator = {
        'x': "x[m]",
        'y': "y[m]",
        'z': "z[m]",
    }

    def get_slice(self, data: np.ndarray, axis: str, index: int) -> np.ndarray:
        """This function returns a slice of the data array along the axis and index provided"""
        assert len(data.shape) == 3, "The full data array must be provided"
        assert axis in ['x', 'y', 'z'], "The axis must be x, y, or z"

        if axis == 'x':
            return data[index, :, :]
        elif axis == 'y':
            return data[:, index, :]
        elif axis == 'z':
            return data[:, :, index]

    def get_slice_from_coordinates(self, data: np.ndarray, axis: str, coordinate: float) -> np.ndarray:
        """This function returns a slice of the data array along the axis and index provided"""
        assert len(data.shape) == 3, "The full data array must be provided"
        assert axis in ['x', 'y', 'z'], "The axis must be x, y, or z"

        axis_full = self.axis_translator[axis]
        # Find closes index of coordinate
        index = np.argmin(np.abs(self.coordinates[axis_full] - coordinate))

        if axis == 'x':
            sliced_data = data[index, :, :]
            # Expand the dimension of the axis to match the data
            sliced_data = np.expand_dims(sliced_data, axis=0)
            return sliced_data

        elif axis == 'y':
            sliced_data = data[:, index, :]
            # Expand the dimension of the axis to match the data
            sliced_data = np.expand_dims(sliced_data, axis=1)
            return sliced_data

        elif axis == 'z':
            sliced_data = data[:, :, index]
            # Expand the dimension of the axis to match the data
            sliced_data = np.expand_dims(sliced_data, axis=2)
            return sliced_data


    def get_shape_dimensions(self, data: np.ndarray) -> tuple:
        """This function returns the cartesians dimensions present on a data array"""
        dims = ''
        if data.shape[0] > 1:
            dims += 'x'
        if data.shape[1] > 1:
            dims += 'y'
        if data.shape[2] > 1:
            dims += 'z'
        return dims


    def axis_centroids(self, axis):
        """This function returns the centroids of the axis provided"""
        assert axis in ['x', 'y', 'z'], "The axis must be x, y, or z"
        axis = self.axis_translator[axis]
        return np.diff(self.coordinates[axis]) + self.coordinates[axis][0:-1]
    @property
    def x_centroid(self):
        return np.diff(self.coordinates['x[m]']) + self.coordinates['x[m]'][0:-1]

    @property
    def y_centroid(self):
        return np.diff(self.coordinates['y[m]']) + self.coordinates['y[m]'][0:-1]

    @property
    def z_centroid(self):
        return np.diff(self.coordinates['z[m]']) + self.coordinates['z[m]'][0:-1]

    @property
    def x_min(self):
        return self.coordinates['x[m]'][0]

    @property
    def x_max(self):
        return self.coordinates['x[m]'][-1]

    @property
    def y_min(self):
        return self.coordinates['y[m]'][0]

    @property
    def y_max(self):
        return self.coordinates['y[m]'][-1]

    @property
    def z_min(self):
        return self.coordinates['z[m]'][0]

    @property
    def z_max(self):
        return self.coordinates['z[m]'][-1]

    @property
    def x_extent(self):
        return self.x_max - self.x_min

    @property
    def y_extent(self):
        return self.y_max - self.y_min

    @property
    def z_extent(self):
        return self.z_max - self.z_min

    @property
    def x_spacing(self):
        return np.diff(self.coordinates['x[m]'])

    @property
    def y_spacing(self):
        return np.diff(self.coordinates['y[m]'])

    @property
    def z_spacing(self):
        return np.diff(self.coordinates['z[m]'])



