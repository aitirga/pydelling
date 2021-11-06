"""
Template to interpolate a set of raster files to a PFLOTRAN mesh
"""
import PyFLOTRAN.utils.globals as globals
import PyFLOTRAN.readers as readers
import PyFLOTRAN.interpolation as interpolation
import PyFLOTRAN.writers as writers

import glob
import numpy as np
import os
import sys


def main(argv):
    # Read configuration file
    # global config
    try:
        config_file_path = argv[1]
    except IndexError as ie:
        config_file_path = "./config.yaml"
        print(f"INFO: Using default config file: {config_file_path}")
        print(f"INFO: You can specify the config file as an argument. " 
              f"Eg: python script/path/to/script_file.py path/to/config_file.yaml")
    globals.initialize_config(config_file=config_file_path)

    # Read centroid files for pressure
    velocity_BC_files_list = glob.glob(globals.config.general.pressure_raster_folder + "/*")
    velocity_BC_files_list = sorted(velocity_BC_files_list)
    output_filename = "top_BC_velocities.h5"
    bc_interpolator = interpolation.SparseDataInterpolator()
    bc_interpolator.remove_output_file(filename=output_filename)
    bc_times = []
    interpolated_array = []
    for year_iterator, year_file in enumerate(velocity_BC_files_list):
        normalized_time = os.path.basename(year_file)
        for split_filter in globals.config.filters.time_from_file_splits:
            normalized_time = normalized_time.split(split_filter[0])[split_filter[1]]
        normalized_time = (float(normalized_time) - globals.config.time.zero_time_modifier) * 365 * 24 * 3600
        print(f"{year_iterator+1} of {len(velocity_BC_files_list)}")
        bc_times.append(normalized_time)
        pressure_raster = readers.CentroidReader(filename=year_file,
                                                 centroid_pos=(1, 3),
                                                 var_pos=6,
                                                 var_name=os.path.basename(year_file),
                                                 header=True)

        bc_interpolator.wipe_data()

        # Convert global coordinates into local
        pressure_raster.global_coords_to_local(x_local_to_global=float(globals.config.coord.x_local_to_global),
                                               y_local_to_global=float(globals.config.coord.y_local_to_global))
        bc_interpolator.add_data(pressure_raster.get_data())
        bc_interpolator.create_regular_mesh(n_x=1000, n_y=1000)
        bc_interpolator.interpolate()
        interpolated_array.append(bc_interpolator.get_data())
    base_writer = writers.HDF5RasterWriter(filename=output_filename, data=np.array(interpolated_array),
                                           info=bc_interpolator.info, times=bc_times)
    base_writer.dump_file(filename=output_filename)


if __name__ == "__main__":
    main(sys.argv)
