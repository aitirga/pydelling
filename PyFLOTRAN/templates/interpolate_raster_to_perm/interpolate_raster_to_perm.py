"""
Template to interpolate a set of raster files to a PFLOTRAN mesh
"""
import os
import PyFLOTRAN.utils.globals as globals
import PyFLOTRAN.readers as readers
import PyFLOTRAN.interpolation as interpolation
from PyFLOTRAN.utils.common import interpolate_permeability_anisotropic
import glob
import numpy as np
def main():
    # Read configuration file
    # global config
    globals.initialize_config(config_file="./config.yaml")
    permX, PFLOTRAN_centroid = interpolate_permeability_anisotropic(
        perm_filename="./data/permeability_data/permX_caseA.dat",
        mesh_filename=globals.config.general.PFLOTRAN_centroid_file
    )
    # PFLOTRAN_centroid = CentroidReader(filename=globals.config.general.PFLOTRAN_centroid_file)
    # PFLOTRAN_centroid = CentroidReader(filename="./data/centroid_mini.dat")
    z_depth = [int(dummy.strip()) for dummy in open("./data/raster_files/z.dat").readlines()]
    raster_file_interpolator = interpolation.SparseDataInterpolator()
    for idx, raster_file in enumerate(glob.glob("./data/raster_files/*.txt")):
        raster_file_data = readers.RasterFileReader(filename=raster_file)
        raster_file_data.add_z_info(z_depth[idx])
        raster_file_interpolator.add_data(raster_file_data.get_data())
    raster_file_interpolator.add_mesh(PFLOTRAN_centroid.get_data())
    raster_file_interpolator.interpolate(method="nearest")

    print(raster_file_interpolator.get_data()[:, 3].min())
    # permX_final = np.multiply(permX.get_data(), raster_file_interpolator.get_data())





if __name__ == "__main__":
    main()
