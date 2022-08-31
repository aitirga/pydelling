"""
Module containing general utility methods
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


def get_config_path():
    from pathlib import Path
    return Path(__file__).parent.parent / "config"


def get_output_path():
    from pathlib import Path
    os.makedirs(Path.cwd() / "output", exist_ok=True)
    return Path.cwd() / "output"


def get_material_id_from_centroids(igp_reader, borehole_coordinates) -> np.array:
    from pydelling.readers.iGPReader.io import iGPReader
    igp_reader: iGPReader = igp_reader
    borehole_coordinates: pd.DataFrame = borehole_coordinates
    centroid_coordinates: np.array = igp_reader.centroids

    borehole_coordinates_values = borehole_coordinates.values[:, 0:3]  # Get numpy array of borehole coordinate values
    material_values = np.empty(shape=igp_reader.centroids.shape[0])
    for material_id, material in enumerate(igp_reader.material_dict):
        material_values[igp_reader.material_dict[material] - 1] = material_id

    interpolated_materials = griddata(centroid_coordinates, material_values, borehole_coordinates_values, method="nearest")
    return interpolated_materials
