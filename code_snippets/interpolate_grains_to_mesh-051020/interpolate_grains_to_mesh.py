"""
Template to interpolate a set of raster files to a PFLOTRAN mesh
"""

import numpy as np
import os
import sys
from PyFLOTRAN.config import config
from PyFLOTRAN.readers import StructuredListReader, OpenFoamReader
from PyFLOTRAN.interpolation import SparseDataInterpolator
from PyFLOTRAN.writers import HDF5CentroidWriter, OpenFoamVariableWriter

def main(argv):
    # Read grain files with the list reader
    grain_reader = StructuredListReader()
    # Read the OpenFOAM mesh
    open_foam_reader = OpenFoamReader()
    # Interpolate the grain information to the OpenFOAM mesh
    sparse_data_interpolator = SparseDataInterpolator(interpolation_data=grain_reader.get_data(),
                                                      mesh_data=open_foam_reader.cell_centers,
                                                      )
    sparse_data_interpolator.interpolate(method="nearest")
    # Change the min value of the eps field to a non-zero value
    sparse_data_interpolator.change_min_value(min_value=config.structured_list_reader.min_value)
    # Write the epsilon field in the OpenFOAM format
    sparse_data_interpolator.write_data(writer_class=OpenFoamVariableWriter)


if __name__ == "__main__":
    main(sys.argv)
