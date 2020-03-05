"""
Template to interpolate a set of raster files to a PFLOTRAN mesh
"""
from PyFLOTRAN.readers import RasterFileReader
import os
import PyFLOTRAN.utils.globals as globals
from PyFLOTRAN.readers.RasterFileReader import RasterFileReader
from PyFLOTRAN.readers.CentroidReader import CentroidReader
from PyFLOTRAN.interpolation.BaseInterpolator import BaseInterpolator
from PyFLOTRAN.interpolation.SparseDataInterpolator import SparseDataInterpolator

def main():
    # Read configuration file
    # global config
    globals.initialize_config(config_file="./config.yaml")
    a = RasterFileReader(filename="S:/PERMEABILITY GLACIAL/Raster File 49100/SHYD_RelK_Layer_01_49100.txt")
    # PFLOTRAN_centroids = CentroidReader(filename="./data/centroid.dat")
    # print(a.get_data())
    a.add_z_info(50.0)
    # print(a.get_data())
    interpo = SparseDataInterpolator(interpolation_data=a.get_data())
    interpo.add_data(a.get_data())
    a.add_z_info(25.0)
    interpo.add_data(a.get_data())
    interpo.add_mesh(a.get_data())
    interpo.interpolate(method="nearest")
    interpo.dump_to_hdf5("proba.h5", var_name="permX")


if __name__ == "__main__":
    main()
