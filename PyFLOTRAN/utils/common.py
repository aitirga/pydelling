"""
Contains general purpose utility functions
"""
import munch
import yaml
# import PyFLOTRAN.readers as readers
# import PyFLOTRAN.interpolation as interpolation
from PyFLOTRAN.readers.CentroidReader import CentroidReader
from PyFLOTRAN.interpolation.SparseDataInterpolator import SparseDataInterpolator

def read_config(config_file="./config.yaml"):
    """
    Reads the configuration file
    :param config_file:
    :return:
    """
    with open(config_file) as file:
        context = yaml.load(file, Loader=yaml.FullLoader)
    return munch.DefaultMunch.fromDict(context)
    # return munch.DefaultMunch.from_dict(context)


def interpolate_permeability_anisotropic(perm_filename, mesh_filename=None, mesh=None):
    perm = CentroidReader(filename=perm_filename, header=True)
    if mesh is None:
        assert mesh_filename is not None, "A mesh file needs to be given under the mesh_filename tag"
        mesh = CentroidReader(filename=mesh_filename, header=False)
    else:
        mesh = mesh
    interpolator = SparseDataInterpolator(interpolation_data=perm.get_data(),
                                          mesh_data=mesh.get_data())
    interpolator.interpolate(method="nearest")
    return interpolator, mesh



