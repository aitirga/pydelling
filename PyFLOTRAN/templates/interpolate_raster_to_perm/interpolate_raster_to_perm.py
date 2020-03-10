"""
Template to interpolate a set of raster files to a PFLOTRAN mesh
"""
import PyFLOTRAN.utils.globals as globals
import PyFLOTRAN.readers as readers
import PyFLOTRAN.interpolation as interpolation
from PyFLOTRAN.utils.common import interpolate_permeability_anisotropic
import glob
import numpy as np
from PyFLOTRAN.writers.HDF5Writer import HDF5Generator


def main():
    # Read configuration file
    # global config

    globals.initialize_config(config_file="./config.yaml")
    perm_folders = glob.glob(globals.config.general.raster_files_folder+"/*RasterFiles*Permafrost*")
    PFLOTRAN_centroid = readers.CentroidReader(filename=globals.config.general.PFLOTRAN_centroid_file, header=False)

    print("Generating Cell IDs")
    cell_IDs = np.arange(1, PFLOTRAN_centroid.info["n_cells"] + 1)
    h5exporter = HDF5Generator(filename="Permeability_interpolated_top_layer.h5")
    # Export Cell IDs
    h5exporter.load_data("Cell Ids", cell_IDs)
    h5exporter.dump_file(remove_if_exists=True)

    #Master permeability files
    permX, PFLOTRAN_centroid = interpolate_permeability_anisotropic(
        perm_filename=globals.config.general.permeability_files_folder + "/permX_newdata.dat",
        mesh=PFLOTRAN_centroid
    )
    # permX.dump_to_csv(filename="permX.csv")

    permY, PFLOTRAN_centroid = interpolate_permeability_anisotropic(
        perm_filename=globals.config.general.permeability_files_folder + "/permY_newdata.dat",
        mesh=PFLOTRAN_centroid
    )
    # permY.dump_to_csv(filename="permY.csv")

    permZ, PFLOTRAN_centroid = interpolate_permeability_anisotropic(
        perm_filename=globals.config.general.permeability_files_folder + "/permZ_newdata.dat",
        mesh=PFLOTRAN_centroid
    )
    # permZ.dump_to_csv(filename="permZ.csv")

    for perm_folder in perm_folders:
        permeability_name_from_folder = perm_folder.split("_")[-1]

        # PFLOTRAN_centroid = CentroidReader(filename="./data/centroid_mini.dat")
        z_depth = [int(dummy.strip()) for dummy in open(globals.config.general.permeability_files_folder+"/z.dat").readlines()]
        raster_file_interpolator = interpolation.SparseDataInterpolator()
        for idx, raster_file in enumerate(glob.glob(perm_folder+"/*.txt")):
            raster_file_data = readers.RasterFileReader(filename=raster_file)
            raster_file_data.add_z_info(z_depth[idx])
            raster_file_data = raster_file_data.get_data()
            raster_file_data[:, 0] -= float(globals.config.coord.x_local_to_global)
            raster_file_data[:, 1] -= float(globals.config.coord.y_local_to_global)
            raster_file_interpolator.add_data(raster_file_data)
        raster_file_interpolator.add_mesh(PFLOTRAN_centroid.get_data())
        raster_file_interpolator.interpolate(method="nearest")
        permX_final = np.multiply(permX.get_data()[:, 3], raster_file_interpolator.get_data()[:, 3])
        print("X permeability generated")
        # Export PermX
        print("Exporting X Permeability")
        h5exporter.load_data(permeability_name_from_folder+"X", permX_final)
        h5exporter.dump_file()

        raster_file_interpolator.wipe_data()
        for idy, raster_file in enumerate(glob.glob(perm_folder+"./*.txt")):
            raster_file_data = readers.RasterFileReader(filename=raster_file)
            raster_file_data.add_z_info(z_depth[idx])
            raster_file_data = raster_file_data.get_data()
            raster_file_data[:, 0] -= float(globals.config.coord.x_local_to_global)
            raster_file_data[:, 1] -= float(globals.config.coord.y_local_to_global)
            raster_file_interpolator.add_data(raster_file_data)
        raster_file_interpolator.add_mesh(PFLOTRAN_centroid.get_data())
        raster_file_interpolator.interpolate(method="nearest")
        permY_final = np.multiply(permY.get_data()[:, 3], raster_file_interpolator.get_data()[:, 3])
        print("Y permeability generated")
        # Export PermY
        print("Exporting Y Permeability")
        h5exporter.load_data(permeability_name_from_folder+"Y", permY_final)
        h5exporter.dump_file()

        raster_file_interpolator.wipe_data()
        for idz, raster_file in enumerate(glob.glob(perm_folder+"/*.txt")):
            raster_file_data = readers.RasterFileReader(filename=raster_file)
            raster_file_data.add_z_info(z_depth[idx])
            raster_file_data = raster_file_data.get_data()
            raster_file_data[:, 0] -= float(globals.config.coord.x_local_to_global)
            raster_file_data[:, 1] -= float(globals.config.coord.y_local_to_global)
            raster_file_interpolator.add_data(raster_file_data)
        raster_file_interpolator.add_mesh(PFLOTRAN_centroid.get_data())
        raster_file_interpolator.interpolate(method="nearest")
        permZ_final = np.multiply(permZ.get_data()[:, 3], raster_file_interpolator.get_data()[:, 3])
        print("Z permeability generated")
        # Export PermZ
        print("Exporting Z Permeability")
        h5exporter.load_data(permeability_name_from_folder+"Z", permZ_final)
        h5exporter.dump_file()

    print("Dumping Done")


if __name__ == "__main__":
    main()
