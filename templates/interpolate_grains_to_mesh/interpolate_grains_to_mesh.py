"""
Template to interpolate a set of raster files to a PFLOTRAN mesh
"""

import numpy as np
import os
import sys
from PyFLOTRAN.config import config
from PyFLOTRAN.readers import StructuredListReader, OpenFoamReader
from PyFLOTRAN.interpolation import SparseDataInterpolator
from PyFLOTRAN.writers import HDF5CentroidWriter


def main(argv):
    # Read configuration file
    # global config
    grain_reader = StructuredListReader()
    open_foam_reader = OpenFoamReader()
    sparse_data_interpolator = SparseDataInterpolator(interpolation_data=grain_reader.get_data(),
                                                      mesh_data=open_foam_reader.cell_centers,
                                                      )
    np.savetxt("openfoam_cc.csv", open_foam_reader.cell_centers, delimiter=",")
    sparse_data_interpolator.interpolate(method="nearest")
    sparse_data_interpolator.write_data()
    # open_foam_reader

if __name__ == "__main__":
    main(sys.argv)
