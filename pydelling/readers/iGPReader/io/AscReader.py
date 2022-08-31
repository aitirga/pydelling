import logging

import h5py
import numpy as np

from pydelling.config import config
from pydelling.readers.iGPReader.io import BaseReader
from pydelling.readers.iGPReader.utils import get_output_path

logger = logging.getLogger(__name__)


class AscReader(BaseReader):
    """
     Class that contains functions to read a rasterized file in .asc format
     """

    def __init__(self, filename):
        self.filename = filename
        self.info_dict = {}
        self.opened_file = open(self.filename, "r")
        self.xydata_computed = False
        self.read_header()
        self.build_data_structure()
        self.read_data()
        logger.debug(f"CSV data has been read from {self.filename}")
        logger.debug(f"{self.info_dict}")

    def read_header(self, n_header=6):
        for i in range(0, n_header):
            line = self.opened_file.readline().split()
            self.info_dict[line[0]] = float(line[1])

    def read_data(self):
        for id, line in enumerate(self.opened_file.readlines()):
            self.data[id] = line.split()

    def build_data_structure(self):
        """
        Function that creates the internal data structure of the raster file
        """
        assert self.info_dict is not {}
        self.data = np.zeros(shape=(int(self.info_dict["nrows"]), int(self.info_dict["ncols"])))
        x_range = np.arange(self.info_dict["xllcorner"], self.info_dict["nrows"] * self.info_dict["cellsize"],
                            self.info_dict["cellsize"])
        y_range = np.arange(self.info_dict["yllcorner"], self.info_dict["ncols"] * self.info_dict["cellsize"],
                            self.info_dict["cellsize"])
        self.x_mesh, self.y_mesh = np.meshgrid(x_range, y_range)
        self.y_mesh = np.flipud(self.y_mesh)  # To fit into the .asc format criteria

    def rebuild_x_y(self):
        x_range = np.arange(self.info_dict["xllcorner"], self.info_dict["nrows"] * self.info_dict["cellsize"],
                            self.info_dict["cellsize"])
        y_range = np.arange(self.info_dict["yllcorner"], self.info_dict["ncols"] * self.info_dict["cellsize"],
                            self.info_dict["cellsize"])
        self.x_mesh, self.y_mesh = np.meshgrid(x_range, y_range)

    def dump_to_xydata(self):
        ndata = int(self.info_dict["nrows"] * self.info_dict["ncols"])
        self.xydata = np.zeros(shape=(ndata, 3))
        x_mesh_flatten = self.x_mesh.flatten()
        y_mesh_flatten = self.y_mesh.flatten()
        for id, data in enumerate(self.data.flatten()):
            self.xydata[id] = (x_mesh_flatten[id], y_mesh_flatten[id], data)
        self.xydata_computed = True
        return self.xydata

    def dump_to_csv(self, output_file):
        """
        Function that writes the ratser data into a csv file
        :param output_file:
        :return:
        """
        logger.info(f"Writing into {output_file}")
        if not self.xydata_computed:
            xydata = self.dump_to_xydata()
        else:
            xydata = self.xydata
        f = open(get_output_path() / output_file, "w")
        for data in xydata:
            if float(data[2]) == -9999.0:
                continue
            f.write(f"{data[0]},{data[1]},{data[2]}\n")
        f.close()

    def dump_to_wsv(self, output_file):
        logger.info(f"Writing into {output_file}")
        if not self.xydata_computed:
            xydata = self.dump_to_xydata()
        else:
            xydata = self.xydata
        f = open(output_file, "w")
        for data in xydata:
            f.write(f"{data[0]} {data[1]} {data[2]}\n")
        f.close()

    def dump_to_asc(self, output_file):
        logger.info(f"Writing into {output_file}")
        file = open(output_file, "w")
        self.write_asc_header(file)
        self.write_asc_data(file)
        file.close()

    def write_asc_header(self, file):
        # assert isinstance(file, type(open)), "is not a correct file"
        for head in self.info_dict:
            file.write(f"{head} {self.info_dict[head]}\n")

    def write_asc_data(self, file):
        np.savetxt(file, self.data, fmt="%3.2f")

    def export_to_gridded_dataset(self, filename=None, attrs=None):
        """This method exports the asc data to the PFLOTRAN gridded dataset hdf5 file format"""
        filename = get_output_path() / filename if filename else get_output_path() / "gridded_dataset.h5"
        growth_factor = config.general.raster_growth_factor if config.general.raster_growth_factor else 1.0
        if config.general.raster_growth_factor:
            logger.warning(f'Scaling the dx and dy values of the gridded dataset by {config.general.raster_growth_factor}')
        with h5py.File(filename, "w") as hdf5_file:
            hdf5_group = hdf5_file.create_group(filename.stem)
            hdf5_dataset = hdf5_group.create_dataset("Data", data=self.data)
            hdf5_group.attrs.create("Dimension", ["XY"], dtype="S3")
            hdf5_group.attrs["Discretization"] = [self.info_dict["cellsize"]*growth_factor, self.info_dict["cellsize"]*growth_factor]
            hdf5_group.attrs["Origin"] = [self.info_dict["xllcorner"], self.info_dict["yllcorner"]]
            hdf5_group.attrs["Cell Centered"] = [True]
            # hdf5_group.attrs["Interpolation Method"] = "STEP"
            # hdf5_group.attrs["Max Buffer Size"] = 1
        logger.info(f"Gridded dataset has been exported at {filename}")

    def change_data(self, data: np.array):
        self.data = data

    def downsample_data(self, slice_factor=2):
        '''
        This module downsamples the data based on a constant stride in each direction
        :param slice_factor: factor to stride the matrix in each dimension

        :return: downsampled dataset
        '''
        self.data = self.data[0::slice_factor, 0::slice_factor]
        self.info_dict["nrows"] = self.data.shape[0]
        self.info_dict["ncols"] = self.data.shape[1]
        self.info_dict["cellsize"] *= slice_factor
        AscReader.rebuild_x_y(self)
        logger.info("Data has been downsampled, these are the new settings:")
        logger.info(f"{self.info_dict}")
