"""
Template to interpolate a set of raster files to a PFLOTRAN mesh
"""
from PyFLOTRAN.readers import RasterFileReader
import os
import PyFLOTRAN.utils.globals as globals
from PyFLOTRAN.readers.RasterFileReader import RasterFileReader
from PyFLOTRAN.readers.CentroidReader import CentroidReader

def main():
    # Read configuration file
    # global config
    globals.initialize_config(config_file="./config.yaml")
    a = RasterFileReader(filename="S:/PERMEABILITY GLACIAL/Raster File 49100/SHYD_RelK_Layer_01_49100.txt")
    centroid = CentroidReader(filename="./")
    # a.add_z_info(50)
    print(a.get_data())
    # Add raster file data into the sparse interpolator
    # for raster_file in os.


if __name__ == "__main__":
    main()
