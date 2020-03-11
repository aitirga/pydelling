"""
Template to interpolate a set of raster files to a PFLOTRAN mesh
"""
import PyFLOTRAN.utils.globals as globals
import PyFLOTRAN.readers as readers
import PyFLOTRAN.interpolation as interpolation
import PyFLOTRAN.writers as writers
from PyFLOTRAN.utils.modelling_utils import interpolate_centroid_to_structured_grid

import glob
import numpy as np
import os


def main():
    # Read configuration file
    # global config
    globals.initialize_config(config_file="./config.yaml")

    # Read centroid files for pressure
    pressure_raster = readers.CentroidReader(filename=os.path.join(globals.config.general.pressure_raster_folder,
                                                                   "hhem_cal22_noditches_rcp45_min_49000_r1_export_velocity.val"),
                                             centroid_pos=(1, 3),
                                             var_pos=6,
                                             # var_name="Vz",
                                             header=True)
    # Convert global coordinates into local
    pressure_raster.global_coords_to_local(x_local_to_global=float(globals.config.coord.x_local_to_global),
                                           y_local_to_global=float(globals.config.coord.y_local_to_global))
    bc_interpolator = interpolation.SparseDataInterpolator(interpolation_data=pressure_raster.get_data())
    bc_interpolator.create_regular_mesh(n_x=1000, n_y=1000)
    bc_interpolator.interpolate()
    bc_interpolator.write_data(writer_class=writers.HDF5RasterWriter,
                               filename="prueba.h5",
                               region_name="top_BC",
                               remove_if_exists=True,
                               )



if __name__ == "__main__":
    main()
