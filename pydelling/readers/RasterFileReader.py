import logging

import numpy as np
from typing import Tuple, List, Union

from .BaseReader import BaseReader

logger = logging.getLogger(__name__)


class RasterFileReader(BaseReader):
    """
    Class that contains functions to read a rasterized file in .asc format
    """
    def __init__(self,
                 filename=None,
                 header=False,
                 read_data=True,
                 **kwargs,):
        super().__init__(filename=filename, header=header, read_data=read_data, **kwargs)

    def open_file(self, filename, n_header=6):
        with open(filename, 'r') as opened_file:
            if self.header:
                opened_file.readline()  # For now, skips the header if it has
            self.read_file(opened_file, n_header)
        self.build_info()

    def read_file(self, opened_file, n_header=6):
        self.read_header(opened_file, n_header=n_header)
        self.build_structure()
        self.read_data(opened_file)
        logger.info(f"Reading ASC raster file from {opened_file}")

    def read_header(self, opened_file, n_header=6):
        flag_stop = False
        while not flag_stop:
            current_position = opened_file.tell()
            line = opened_file.readline().split()
            if len(line) != 2:
                opened_file.seek(current_position)
                flag_stop = True

            self.info["reader"][line[0]] = float(line[1])
        if 'cellsize' in self.info["reader"].keys():
            self.info["reader"]["dx"] = self.info["reader"]["cellsize"]
            self.info["reader"]["dy"] = self.info["reader"]["cellsize"]


    def read_data(self, opened_file):
        for id, line in enumerate(opened_file.readlines()):
            self.data[id] = np.array(line.split(), dtype=np.float32)

    def build_structure(self):
        assert self.info is not {}
        self.xydata_computed = False
        self.data = np.zeros(shape=(int(self.reader_info["nrows"]), int(self.reader_info["ncols"])))

        if 'cellsize' in self.reader_info:
            x_range = np.arange(self.reader_info["xllcorner"], self.reader_info["xllcorner"] + self.reader_info["nrows"] * self.reader_info["cellsize"],
                                self.reader_info["cellsize"])
            y_range = np.arange(self.reader_info["yllcorner"], self.reader_info["yllcorner"] + self.reader_info["ncols"] * self.reader_info["cellsize"],
                                self.reader_info["cellsize"])
            print(self.reader_info['xllcorner'], self.reader_info['yllcorner'], self.reader_info['cellsize'], self.reader_info['nrows'], self.reader_info['ncols'])
        else:
            x_range = np.arange(self.reader_info["xllcorner"], self.reader_info["xllcorner"] + self.reader_info["nrows"] * self.reader_info["dx"],
                                self.reader_info["dx"])
            y_range = np.arange(self.reader_info["yllcorner"], self.reader_info["yllcorner"] + self.reader_info["ncols"] * self.reader_info["dy"],
                                self.reader_info["dy"])
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
        # self.rebuild_x_y()
        if hasattr(self, "z_coord"):
            return self.get_xyz_data()
        else:
            return self.get_xy_data()

    def rebuild_x_y(self):
        if 'cellsize' in self.info['reader']:
            x_range = np.arange(self.info['reader']["xllcorner"],
                                self.info['reader']["xllcorner"] + self.info['reader']["nrows"] * self.info['reader']["cellsize"],
                                self.info['reader']["cellsize"])
            y_range = np.arange(self.info['reader']["yllcorner"],
                                self.info['reader']["yllcorner"] + self.info['reader']["ncols"] * self.info['reader']["cellsize"],
                                self.info['reader']["cellsize"])
        else:
            x_range = np.arange(self.info['reader']["xllcorner"],
                                self.info['reader']["xllcorner"] + self.info['reader']["nrows"] * self.info['reader']["dx"],
                                self.info['reader']["dx"])
            y_range = np.arange(self.info['reader']["yllcorner"],
                                self.info['reader']["yllcorner"] + self.info['reader']["ncols"] * self.info['reader']["dy"],
                                self.info['reader']["dy"])
        self.x_mesh, self.y_mesh = np.meshgrid(x_range, y_range)
        self.y_mesh = np.flipud(self.y_mesh)  # To fit into the .asc format criteria


    def get_xy_data(self) -> np.ndarray:
        ndata = int(self.info["reader"]["nrows"] * self.info["reader"]["ncols"])
        self.xydata = np.zeros(shape=(ndata, 3))
        rows_mesh_flatten = np.fliplr(self.x_mesh).T.flatten()
        cols_mesh_flatten = np.flipud(self.y_mesh).T.flatten()
        for id, data in enumerate(self.data.flatten()):
            self.xydata[id] = (cols_mesh_flatten[id], rows_mesh_flatten[id], data)
        self.xydata_computed = True
        return self.xydata

    def get_xyz_data(self) -> np.ndarray:
        assert hasattr(self, "z_coord"), "The z-coordinate of this raster file is not given"
        ndata = int(self.info["reader"]["nrows"] * self.info["reader"]["ncols"])
        self.flatten_data = np.zeros(shape=(ndata, 4))
        rows_mesh_flatten = np.fliplr(self.x_mesh).T.flatten()
        cols_mesh_flatten = np.flipud(self.y_mesh).T.flatten()
        for id, data in enumerate(self.data.flatten()):
            self.flatten_data[id] = (cols_mesh_flatten[id], rows_mesh_flatten[id], self.z_coord, data)
        self.xydata_computed = True
        return self.flatten_data

    def to_csv(self, output_file, z_coord=None):
        """
        Function that writes the ratser data into a csv file
        :param output_file:
        :return:
        """

        xydata = self.get_xy_data()
        f = open(output_file, "w")
        if z_coord is not None:
            f.write(f"x,y,z,data\n")
        else:
            f.write(f"x,y,data\n")
        for data in xydata:
            if z_coord is not None:
                f.write(f"{data[0]},{data[1]},{z_coord},{data[2]}\n")
            else:
                f.write(f"{data[0]},{data[1]},{data[2]}\n")
        f.close()
        logger.info(f"The raster file points have been exported to the CSV file {output_file}")

    def to_wsv(self, output_file):
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

    def to_asc(self, output_file):
        logger.info(f"Starting dump into {output_file}")
        with open(output_file, 'w') as file:
            self.write_asc_header(file)
            self.write_asc_data(file)

    def write_asc_header(self, file):
        # assert isinstance(file, type(open)), "is not a correct file"
        # Write info in ASC format
        file.write(f"ncols {self.info['reader']['ncols']}\n")
        file.write(f"nrows {self.info['reader']['nrows']}\n")
        if 'cellsize' in self.info:
            file.write(f"cellsize {self.info['reader']['cellsize']}\n")
        else:
            file.write(f"dx {self.info['reader']['dx']}\n")
            file.write(f"dy {self.info['reader']['dy']}\n")
        file.write(f"xllcorner {self.info['reader']['xllcorner']}\n")
        file.write(f"yllcorner {self.info['reader']['yllcorner']}\n")
        file.write(f"NODATA_value {self.info['reader']['NODATA_value']}\n")


    def write_asc_data(self, file):
        np.savetxt(file, self.data)

    def downsample_data(self, slice_factor=2):
        '''
        This module downsamples the data based on a constant stride in each direction
        :param slice_factor: factor to stride the matrix in each dimension

        :return: downsampled dataset
        '''
        self.data = self.data[0::slice_factor, 0::slice_factor]
        self.info['reader']["nrows"] = self.data.shape[0]
        self.info['reader']["ncols"] = self.data.shape[1]
        if "cellsize" in self.info["reader"]:
            self.info['reader']["cellsize"] *= slice_factor
        else:
            self.info['reader']["dx"] *= slice_factor
            self.info['reader']["dy"] *= slice_factor
        print(self.data.shape)
        self.rebuild_x_y()
        logger.info(f"Data has been downsampled by a factor of {slice_factor}")

    def get_value_from_coord(self, x: float, y: float):
        """
        This method returns the raster value finding the nearest neighbour of a given x, y coordinate
        Args:
            x: Coordinate of the x-direction
            y: Coordinate of the y-direction

        Returns:
            Raster value at the given coordinate
        """
        # Define auxiliaty variables
        d_raster = self.reader_info["cellsize"]
        origin_x = self.reader_info["xllcorner"]
        origin_y = self.reader_info["yllcorner"]

        ix = int(np.floor((x - origin_x) / (1.001 * d_raster)))  # 1.001 value is used to avoid issues with
        # the floor function
        iy = int(np.floor((y - origin_y) / (1.001 * d_raster)))  # 1.001 value is used to avoid issues with
        # the floor function
        iy = int(self.reader_info["ncols"] - iy - 1)
        return self.data[iy, ix]

    def flip_y(self):
        self.info['reader']["yllcorner"] = self.info['reader']["yllcorner"] + self.info['reader']["cellsize"] * self.info['reader']["nrows"]
        RasterFileReader.rebuild_x_y(self)

    def flip_x(self):
        self.info['reader']["xllcorner"] = self.info['reader']["xllcorner"] + self.info['reader']["cellsize"] * self.info['reader']["ncols"]
        RasterFileReader.rebuild_x_y(self)

    def get_plot_image(self, ax=None,
                       fig=None,
                       colorbar=True,
                       colorbar_label=None,
                       **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(self.data, **kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if colorbar:
            fig.colorbar(ax.get_images()[0], label=colorbar_label)
        return ax

    def plot(self, colorbar=True, colorbar_label=None, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        self.get_plot_image(ax=ax,
                            fig=fig,
                            colorbar=colorbar,
                            colorbar_label=colorbar_label,
                            **kwargs)
        plt.show()

    def save_plot(self, output_file, colorbar=None, colorbar_label=None,**kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        self.get_plot_image(ax=ax,
                            fig=fig,
                            colorbar=colorbar,
                            colorbar_label=colorbar_label,
                            **kwargs)
        plt.savefig(output_file)

    @property
    def nx(self) -> int:
        return int(self.info['reader']["ncols"])

    @property
    def ny(self) -> int:
        return int(self.info['reader']["nrows"])

    @property
    def nrows(self) -> int:
        return int(self.info['reader']["nrows"])

    @property
    def ncols(self) -> int:
        return int(self.info['reader']["ncols"])

    @property
    def dx(self):
        if 'dx' in self.info['reader']:
            return self.info['reader']['dx']
        else:
            return self.info['reader']["cellsize"]

    @property
    def dy(self):
        if 'dy' in self.info['reader']:
            return self.info['reader']['dy']
        else:
            return self.info['reader']["cellsize"]

    # Read raster file from a .csv file
    @classmethod
    def from_xyz_csv(self, file_name, dims=None,):
        data = np.loadtxt(file_name, delimiter=",", skiprows=1)
        if dims is None:
            dims = self._guess_raster_dimensions_from_data(data)
        else:
            assert dims[0] * dims[1] == data.shape[0], "The dimensions are not correct"
            dims = (dims[1], dims[0])

        info = {'reader': {}}
        info['reader']["ncols"] = dims[1]
        info['reader']["nrows"] = dims[0]
        info['reader']["xllcorner"] = data[:, 0].min()
        info['reader']["yllcorner"] = data[:, 1].min()
        info['reader']['dx'] = data[1, 0] - data[0, 0]
        info['reader']['dy'] = data[dims[1], 1] - data[0, 1]
        info['reader']['dx'] = info['reader']['dx']
        info['reader']['dy'] = - info['reader']['dy']
        info['reader']["NODATA_value"] = -9999
        return_raster = RasterFileReader(filename=file_name, data=data, info=info, read_data=False)
        return_raster.build_structure()
        return_raster.data = data[:, 2].reshape(dims[0], dims[1])
        return return_raster

    @staticmethod
    def _guess_raster_dimensions_from_data(data: np.ndarray) -> Tuple[int, int]:
        """
        This method guesses the raster dimensions from the data
        Args:
            data: array with the raster data

        Returns:
            Tuple with the number of rows and columns
        """
        if data.shape[0] == data.shape[1]:
            # Square raster
            return data.shape[0], data.shape[1]
        else:
            # Assume dimensions are ordered in x, y, target
            first_x_value = data[0, 0]
            first_y_value = data[0, 1]
            steps_with_same_x = 1
            steps_with_same_y = 1

            for idx in range(1, data.shape[0]):
                if data[idx, 0] == first_x_value:
                    steps_with_same_x += 1
                if data[idx, 1] == first_y_value:
                    steps_with_same_y += 1
            assert steps_with_same_x * steps_with_same_y == data.shape[0], f"Cant guess dimensions from data. Computed dimensions are {steps_with_same_x}x{steps_with_same_y}={steps_with_same_x * steps_with_same_y} and data has {data.shape[0]} elements"
            return steps_with_same_x, steps_with_same_y

    def get_data_from_coordinates(self, x, y):
        """
        This method returns the data from the raster at the given coordinates
        Args:
            x: x coordinate
            y: y coordinate

        Returns:
            The value of the raster at the given coordinates
        """
        x_idx = np.argmin(np.abs(self.y - y))
        y_idx = np.argmin(np.abs(self.x - x))
        return self.data[self.nrows - x_idx - 1, y_idx]

    @property
    def x(self):
        return self.info['reader']["xllcorner"] + np.arange(self.nx) * self.dx

    @property
    def y(self):
        return self.info['reader']["yllcorner"] + np.arange(self.ny) * self.dy

    # Add difference of two raster files
    def __sub__(self, other):
        if isinstance(other, RasterFileReader):
            new_raster = RasterFileReader(filename=self.filename, read_data=False, info=self.info, data=self.data - other.data)
            for key in self.__dict__.keys():
                if key != "data" and key != "info":
                    new_raster.__dict__[key] = self.__dict__[key]
            return new_raster
        elif isinstance(other, (int, float)):
            new_raster = RasterFileReader(filename=self.filename, read_data=False, info=self.info, data=self.data - other)
            for key in self.__dict__.keys():
                if key != "data" and key != "info":
                    new_raster.__dict__[key] = self.__dict__[key]

            return new_raster
        else:
            raise ValueError("The other object is not a RasterFileReader or a number")

    # Add sum of two raster files
    def __add__(self, other):
        if isinstance(other, RasterFileReader):
            new_raster = RasterFileReader(filename=self.filename, read_data=False, info=self.info, data=self.data + other.data)
            for key in self.__dict__.keys():
                if key != "data" and key != "info":
                    new_raster.__dict__[key] = self.__dict__[key]

            return new_raster
        elif isinstance(other, (int, float)):
            new_raster = RasterFileReader(filename=self.filename, read_data=False, info=self.info, data=self.data + other)
            for key in self.__dict__.keys():
                if key != "data" and key != "info":
                    new_raster.__dict__[key] = self.__dict__[key]

            return new_raster
        else:
            raise ValueError("The other object is not a RasterFileReader or a number")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_raster = RasterFileReader(filename=self.filename, read_data=False, info=self.info, data=self.data * other)
            for key in self.__dict__.keys():
                if key != "data" and key != "info":
                    new_raster.__dict__[key] = self.__dict__[key]
            return new_raster

        elif isinstance(other, RasterFileReader):
            new_raster = RasterFileReader(filename=self.filename, read_data=False, info=self.info, data=self.data * other.data)
            for key in self.__dict__.keys():
                if key != "data" and key != "info":
                    new_raster.__dict__[key] = self.__dict__[key]

            return new_raster

        else:
            raise ValueError("The other object is not a RasterFileReader or a number")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            new_raster = RasterFileReader(filename=self.filename, read_data=False, info=self.info, data=self.data / other)
            for key in self.__dict__.keys():
                if key != "data" and key != "info":
                    new_raster.__dict__[key] = self.__dict__[key]
            return new_raster

        elif isinstance(other, RasterFileReader):
            new_raster = RasterFileReader(filename=self.filename, read_data=False, info=self.info, data=self.data / other.data)
            for key in self.__dict__.keys():
                if key != "data" and key != "info":
                    new_raster.__dict__[key] = self.__dict__[key]

            return new_raster

        else:
            raise ValueError("The other object is not a RasterFileReader or a number")

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            new_raster = RasterFileReader(filename=self.filename, read_data=False, info=self.info, data=self.data ** other)
            for key in self.__dict__.keys():
                if key != "data" and key != "info":
                    new_raster.__dict__[key] = self.__dict__[key]
            return new_raster

        elif isinstance(other, RasterFileReader):
            new_raster = RasterFileReader(filename=self.filename, read_data=False, info=self.info, data=self.data ** other.data)
            new_raster.info = self.info
            for key in self.__dict__.keys():
                if key != "data" and key != "info":
                    new_raster.__dict__[key] = self.__dict__[key]


            return new_raster

        else:
            raise ValueError("The other object is not a RasterFileReader or a number")

    @property
    def reader_info(self):
        return self.info['reader']










