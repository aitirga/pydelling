import numpy as np
import pandas as pd
import h5py

class PflotranProcessingUtils:
    """This class contains utility functions to use on the PflotranReader class"""
    variables: list
    coordinates: np.ndarray
    data: h5py.File
    # data:

    def get_slice(self, data: np.ndarray, axis: int, index: int) -> np.ndarray:
        """This function returns a slice of the data array along the axis and index provided"""
        return data.take(index, axis=axis)


    @property
    def x_centroid(self):
        return np.diff(self.coordinates['x[m]']) + self.coordinates['x[m]'][0:-1]

    @property
    def y_centroid(self):
        return np.diff(self.coordinates['y[m]']) + self.coordinates['y[m]'][0:-1]

    @property
    def z_centroid(self):
        return np.diff(self.coordinates['z[m]']) + self.coordinates['z[m]'][0:-1]

