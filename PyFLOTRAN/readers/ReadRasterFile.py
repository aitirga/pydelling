class ReadRasterFile:
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
        print("Data has been read, these are the settings:")
        print(self.info_dict)

    def read_header(self, n_header=6):
        for i in range(0, n_header):
            line = self.opened_file.readline().split()
            self.info_dict[line[0]] = float(line[1])

    def read_data(self):
        print(f"Reading data from {self.filename}")
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
        for head in self.info_dict:
            file.write(f"{head} {self.info_dict[head]}\n")

    def write_asc_data(self, file):
        np.savetxt(file, self.data, fmt="%3.2f")

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
        print("Data has been downsampled, the new settings are these:")
        print(self.info_dict)