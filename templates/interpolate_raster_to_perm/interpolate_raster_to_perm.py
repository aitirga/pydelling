"""
Template to interpolate a set of raster files to a PFLOTRAN mesh
"""
import PyFLOTRAN.utils.globals as globals
import PyFLOTRAN.readers as readers
import PyFLOTRAN.interpolation as interpolation
from PyFLOTRAN.utils.modelling_utils import interpolate_permeability_anisotropic
from PyFLOTRAN.writers.HDF5CentroidWriter import HDF5CentroidWriter
import glob
import numpy as np
import os



def main():
    # Read configuration file
    # global config
    globals.initialize_config(config_file="./config.yaml")

    perm_folders = glob.glob(globals.config.general.raster_files_folder+"/Perm*")
    print(perm_folders)
    PFLOTRAN_centroid = readers.CentroidReader(filename=globals.config.general.PFLOTRAN_centroid_file, header=False)
    # normal_range = np.arange(1, PFLOTRAN_centroid.info["n_cells"] + 1)
    # diff_array = PFLOTRAN_centroid.get_data()[:, 3] - normal_range
    # diff_array[diff_array != 0.0] = 1.0
    # print(np.sum(diff_array))
    print("Generating Cell IDs")
    # cell_IDs = np.arange(1, PFLOTRAN_centroid.info["n_cells"] + 1)
    h5exporter = HDF5CentroidWriter(filename="Permeability_interpolated_top_layer.h5")
    h5exporter.remove_output_file()
    # Export Cell IDs
    h5exporter.load_data("Cell Ids", np.array(PFLOTRAN_centroid.get_data()[:, 3], dtype=np.int32))
    h5exporter.dump_file()

    #Master permeability files
    permX = interpolate_permeability_anisotropic(
        perm_filename=globals.config.general.permeability_files_folder + "/permX_newdata.dat",
        mesh=PFLOTRAN_centroid
    )
    # permX.dump_to_csv(filename="permX-original.csv")

    permY = interpolate_permeability_anisotropic(
        perm_filename=globals.config.general.permeability_files_folder + "/permY_newdata.dat",
        mesh=PFLOTRAN_centroid
    )
    # permY.dump_to_csv(filename="permY-original.csv")
    permZ = interpolate_permeability_anisotropic(
        perm_filename=globals.config.general.permeability_files_folder + "/permZ_newdata.dat",
        mesh=PFLOTRAN_centroid
    )
    # permZ.dump_to_csv(filename="permZ.csv")

    for stepID, perm_folder in enumerate(perm_folders):
        print(f"Step {stepID} of {len(perm_folders)}")
        permeability_name_from_folder = os.path.basename(perm_folder)

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
        temp_array = np.reshape(permX_final, (permX_final.shape[0], 1))
        temp_array = np.concatenate((permX.mesh, temp_array), axis=1)
        # np.savetxt("permX.csv", temp_array, delimiter=",")

        # Export PermX
        print("Exporting X Permeability")
        h5exporter.load_data(permeability_name_from_folder+"X", permX_final)
        h5exporter.dump_file()


        permY_final = np.multiply(permY.get_data()[:, 3], raster_file_interpolator.get_data()[:, 3])
        print("Y permeability generated")
        # Export PermY
        print("Exporting Y Permeability")
        h5exporter.load_data(permeability_name_from_folder+"Y", permY_final)
        h5exporter.dump_file()


        permZ_final = np.multiply(permZ.get_data()[:, 3], raster_file_interpolator.get_data()[:, 3])
        print("Z permeability generated")
        # Export PermZ
        print("Exporting Z Permeability")
        h5exporter.load_data(permeability_name_from_folder+"Z", permZ_final)
        h5exporter.dump_file()
        # np.savetxt("permZ.csv", temp_array, delimiter=",")

    print("Dumping Done")


if __name__ == "__main__":
    main()
