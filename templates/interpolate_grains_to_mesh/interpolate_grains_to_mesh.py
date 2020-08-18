"""
Template to interpolate a set of raster files to a PFLOTRAN mesh
"""

import numpy as np
import os
import sys
from PyFLOTRAN.config import config
from PyFLOTRAN.readers import StructuredListReader

def main(argv):
    # Read configuration file
    # global config
    grain_reader = StructuredListReader()
    print(grain_reader.get_data())
    print(grain_reader.values)


if __name__ == "__main__":
    main(sys.argv)
