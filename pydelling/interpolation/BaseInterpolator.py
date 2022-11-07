"""
This class implements a basic interface for Interpolation classes
"""
import logging
import os

import h5py
import numpy as np
import pandas as pd
import seaborn as sns

from pydelling.config import config
from pydelling.utils.decorators import set_run
from ..writers.BaseWriter import BaseWriter

logger = logging.getLogger(__name__)



class BaseInterpolator:
    is_run: bool = False
    has_regular_mesh: bool = False

    def __init__(self,
                 interpolation_data=None,
                 mesh_data=None):
        self.data: np.ndarray = []
        self.mesh: np.ndarray = []
        self.info = {"interpolation": {}}
        self.interpolated_data: np.ndarray = []
        self.filename = None
        if interpolation_data is not None:
            self.add_data(data=interpolation_data)
        if mesh_data is not None:
            self.add_mesh(data=mesh_data)

    def add_data(self, data):
        """
        Add a dataset that will be used to interpolate
        :return:
        """
        if len(self.data) == 0:
            self.data = np.array(data)
        else:
            self.data = np.vstack((self.data, data))

    def add_mesh(self, data, id_index=3):
        """
        Add the set of points on which interpolation will be performed
        :return:
        """
        if data.shape[1] > 3:
            temp_id_data = data[:, id_index]
            temp_data = data[:, 0:3]
        else:
            temp_data = data

        if not self.mesh:
            self.mesh = np.array(temp_data)
            if data.shape[1] > 3:
                self.id_data = np.array(temp_id_data)
        else:
            self.mesh = np.vstack((self.mesh, temp_data))
            if data.shape[1] > 3:
                self.id_data = np.vstack((self.id_data, temp_id_data))

    @set_run
    def run(self):
        """
        Runs the interpolation algorithm
        Returns:
            A numpy array containing the values of the interpolated variable on the mesh centroids
        """
        self.interpolated_data = self.mesh[:, 0]
        return self.interpolated_data

    def get_data(self):
        """
        Returns interpolated data
        :return:
        """
        return self.interpolated_data

    def dump_to_hdf5(self, filename=None, var_name=None, data=None):
        """
        Dumps the data into HDF5 format
        :return:
        """
        if data is None:
            data = self.interpolated_data
        if filename is not None:
            self.filename = filename
        if not os.path.exists(filename):
            tempfile = h5py.File(filename, "w")
            tempfile.close()
        with h5py.File(filename, "r+") as tempfile:
            tempfile.create_dataset(var_name, data=data)

    def dump_to_csv(self, filename=None, **kwargs):
        temp_array = np.reshape(self.interpolated_data, (self.interpolated_data.shape[0], 1))
        temp_array = np.concatenate((self.mesh, temp_array), axis=1)
        np.savetxt(filename, temp_array, **kwargs)
        if config.general.verbose:
            print(f"Data has been dumped into {filename}")

        # return np.concatenate((self.mesh, temp_array), axis=1)

    def wipe_data(self):
        """Wipes data structure
        """
        self.data = []
        self.mesh = []
        self.interpolated_data = []

    def write_data(self, writer_class=BaseWriter, filename=None, **kwargs):
        base_writer = writer_class(filename=filename, data=self.get_data(), info=self.info, **kwargs)
        base_writer.run(filename=filename)

    def remove_output_file(self, writer_class=BaseWriter, filename=None, **kwargs):
        base_writer = writer_class(filename=filename, **kwargs)
        base_writer.remove_output_file(filename)

    def get_minmax_coords(self):
        self.data_xmin = np.min(self.data[:, 0])
        self.data_xmax = np.max(self.data[:, 0])
        self.data_ymin = np.min(self.data[:, 1])
        self.data_ymax = np.max(self.data[:, 1])
        self.info["interpolation"]["x_min"] = self.data_xmin
        self.info["interpolation"]["x_max"] = self.data_xmax
        self.info["interpolation"]["y_min"] = self.data_ymin
        self.info["interpolation"]["y_max"] = self.data_ymax

    def create_regular_mesh(self, n_x, n_y, dilatation_factor=1.0):
        """Create an inner regular mesh"""
        self.has_regular_mesh = True
        self.mesh = []
        self.get_minmax_coords()
        dx = abs(self.data_xmax - self.data_xmin) / n_x
        dx_dil = dx * dilatation_factor
        dx_correction = (dx_dil - dx) / 2.0

        dy = abs(self.data_ymax - self.data_ymin) / n_y
        dy_dil = dy * dilatation_factor
        dy_correction = (dy_dil - dy) / 2.0

        self.info["interpolation"].update({"n_x": n_x,
                                           "n_y": n_y,
                                           "dilatation_factor": dilatation_factor,
                                           "type": "regular_mesh",
                                           "d_x": dx_dil,
                                           "d_y": dy_dil,
                                           })

        linspace_x = np.linspace(self.data_xmin - dx_correction, self.data_xmax + dx_correction, n_x)
        linspace_y = np.linspace(self.data_ymin - dy_correction, self.data_ymax + dy_correction, n_y)
        grid_x, grid_y = np.meshgrid(linspace_x, linspace_y)
        self.mesh = np.hstack((grid_x.reshape((grid_x.size, 1)), grid_y.reshape((grid_y.size, 1))))

    def describe(self, write_to_file=None, plots=True):
        assert self.is_run, "The interpolator has not been run"
        temp_df = pd.DataFrame(self.interpolated_data)
        logger.info("Describing the interpolated data")
        print(temp_df.describe())
        if write_to_file:
            if type(write_to_file) is str:
                temp_df.describe().to_csv(write_to_file)
            else:
                temp_df.describe().to_csv("interpolated_data-description.csv")
            logger.info("Writing the description to file")
        if plots:
            logger.info("Plotting data")
            import matplotlib.pyplot as plt
            histogram_plot = sns.kdeplot(x=self.interpolated_data)
            plt.show()




