from PyFLOTRAN.config import config
from PyFLOTRAN.readers import iGPReader


def main():
    read_iGP = iGPReader(config.general.igp_project_folder,
                                 build_mesh=False,
                                 project_name=config.general.project_name,
                                 output_folder="./output/",
                         )
    # Write the initial mesh into a .csv for post-processing in Paraview
    # First, build a mathematical structure of the mesh
    read_iGP.build_mesh_data()
    read_iGP.implicit_to_explicit()
    read_iGP.write_hdf5_domain()


if __name__ == "__main__":
    main()
