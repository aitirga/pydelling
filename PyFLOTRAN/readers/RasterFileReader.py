import numpy as np
from .BaseReader import BaseReader
from ..utils import globals


class RasterFileReader(BaseReader):
    """
    Class that contains functions to read a rasterized file in .asc format
    """
    def read_file(self, opened_file):
        self.read_header(opened_file)
        self.build_structure()
        self.read_data(opened_file)

    def read_header(self, opened_file, n_header=6):
        for i in range(0, n_header):
            line = opened_file.readline().split()
            self.info["reader"][line[0]] = float(line[1])

    def read_data(self, opened_file):
        if globals.config.general.verbose:
            print(f"Reading data from {self.filename}")
        for id, line in enumerate(opened_file.readlines()):
            self.data[id] = np.array(line.split(), dtype=np.float32)

    def build_structure(self):
        assert self.info is not {}
        self.reader_info = self.info["reader"]
        self.xydata_computed = False
        self.data = np.zeros(shape=(int(self.reader_info["nrows"]), int(self.reader_info["ncols"])))
        x_range = np.arange(self.reader_info["xllcorner"], self.reader_info["xllcorner"] + self.reader_info["nrows"] * self.reader_info["cellsize"],
                            self.reader_info["cellsize"])
        y_range = np.arange(self.reader_info["yllcorner"], self.reader_info["yllcorner"] + self.reader_info["ncols"] * self.reader_info["cellsize"],
                            self.reader_info["cellsize"])
        self.x_mesh, self.y_mesh = np.meshgrid(x_range, y_range)
        self.y_mesh = np.flipud(self.y_mesh)  # To fit into the .asc format criteria

    def build_info(self):
        """
        Function that creates the internal data structure of the raster file
        """
        self.info["reader"]["filename"] = self.filename

    def add_z_info(self, z_coord):
        self.z_coord = z_coord

    def get_data(self) -> np.ndarray:
        if hasattr(self, "z_coord"):
            return self.get_xyz_data()
        else:
            return self.get_xy_data()

    def rebuild_x_y(self):
        x_range = np.arange(self.info["reader"]["xllcorner"], self.info["reader"]["nrows"] * self.info["reader"]["cellsize"],
                            self.info["reader"]["cellsize"])
        y_range = np.arange(self.info["reader"]["yllcorner"], self.info["reader"]["ncols"] * self.info["reader"]["cellsize"],
                            self.info["reader"]["cellsize"])
        self.x_mesh, self.y_mesh = np.meshgrid(x_range, y_range)

    def get_xy_data(self) -> np.ndarray:
        ndata = int(self.info["reader"]["nrows"] * self.info["reader"]["ncols"])
        self.xydata = np.zeros(shape=(ndata, 3))
        x_mesh_flatten = self.x_mesh.flatten()
        y_mesh_flatten = self.y_mesh.flatten()
        for id, data in enumerate(self.data.flatten()):
            self.xydata[id] = (x_mesh_flatten[id], y_mesh_flatten[id], data)
        self.xydata_computed = True
        return self.xydata

    def get_xyz_data(self) -> np.ndarray:
        assert hasattr(self, "z_coord"), "The z-coordinate of this raster file is not given"
        ndata = int(self.info["reader"]["nrows"] * self.info["reader"]["ncols"])
        self.flatten_data = np.zeros(shape=(ndata, 4))
        x_mesh_flatten = self.x_mesh.flatten()
        y_mesh_flatten = self.y_mesh.flatten()
        for id, data in enumerate(self.data.flatten()):
            self.flatten_data[id] = (x_mesh_flatten[id], y_mesh_flatten[id], self.z_coord, data)
        self.xydata_computed = True
        return self.flatten_data

    def dump_to_csv(self, output_file):
        """
        Function that writes the ratser data into a csv file
        :param output_file:
        :return:
        """
        print(f"Starting dump into {output_file}")
        if not self.xydata_computed:
            xydata = self.dump_to_xydata()
        else:
            xydata = self.xydata
        f = open(output_file, "w")
        for data in xydata:
            f.write(f"{data[0]},{data[1]},{data[2]}\n")
        f.close()
        print(f"The data has been properly exported to the {output_file} file")

    def dump_to_wsv(self, output_file):
        print(f"Starting dump into {output_file}")
        if not self.xydata_computed:
            xydata = self.dump_to_xydata()
        else:
            xydata = self.xydata
        f = open(output_file, "w")
        for data in xydata:
            f.write(f"{data[0]} {data[1]} {data[2]}\n")
        f.close()
        print(f"The data has been properly exported to the {output_file} file")

    def dump_to_asc(self, output_file):
        print(f"Starting dump into {output_file}")
        file = open(output_file, "w")
        self.write_asc_header(file)
        self.write_asc_data(file)
        file.close()

    def write_asc_header(self, file):
        # assert isinstance(file, type(open)), "is not a correct file"
        for head in self.info:
            file.write(f"{head} {self.info[head]}\n")

    def write_asc_data(self, file):
        np.savetxt(file, self.data, fmt="%3.2f")

    def downsample_data(self, slice_factor=2):
        '''
        This module downsamples the data based on a constant stride in each direction
        :param slice_factor: factor to stride the matrix in each dimension

        :return: downsampled dataset
        '''
        self.data = self.data[0::slice_factor, 0::slice_factor]
        self.info["nrows"] = self.data.shape[0]
        self.info["ncols"] = self.data.shape[1]
        self.info["cellsize"] *= slice_factor
        RasterFileReader.rebuild_x_y(self)
        print("Data has been downsampled, the new settings are these:")
        print(self.info)

