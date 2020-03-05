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


def interpolate_permeability_anisotropic(perm_filename, mesh_filename):
    perm = CentroidReader(filename=perm_filename, header=True)
    mesh = CentroidReader(filename=mesh_filename, header=False)
    interpolator = SparseDataInterpolator(interpolation_data=perm.get_data(),
                                          mesh_data=mesh.get_data())
    interpolator.interpolate(method="nearest")
    return interpolator, mesh



