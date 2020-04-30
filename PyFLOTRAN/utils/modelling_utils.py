"""
Contains general purpose utility functions
"""
import munch
import yaml
import PyFLOTRAN.readers as readers
import PyFLOTRAN.interpolation as interpolation
import numpy as np


def interpolate_permeability_anisotropic(perm_filename, mesh_filename=None, mesh=None):
    perm = readers.CentroidReader(filename=perm_filename, header=True)
    if mesh is None:
        assert mesh_filename is not None, "A mesh file needs to be given under the mesh_filename tag"
        mesh = readers.CentroidReader(filename=mesh_filename, header=False)
    else:
        mesh = mesh
    interpolator = interpolation.SparseDataInterpolator(interpolation_data=perm.get_data(),
                                          mesh_data=mesh.get_data())
    interpolator.interpolate(method="nearest")
    if mesh is None:
        return interpolator, mesh
    else:
        return interpolator


def interpolate_centroid_to_structured_grid(centroid: np.ndarray,
                                            var: np.ndarray,
                                            automatic=True,
                                            n_x=100,
                                            n_y=100,
                                            origin_x=0.0,
                                            final_x=1.0,
                                            origin_y=0.0,
                                            final_y=1.0) -> np.ndarray:
    """Reads a set of 2D centroid points and interpolates them into a structured grid"""
    # Check that the centroid file is a 2D array
    assert centroid.shape[1] == 2, "The given centroid file is not a 2D array"
    _var = var
    _centroid = centroid
    _dx = abs(final_x - origin_x)/n_x
    _dy = abs(final_y - origin_y) / n_y
    linspace_x = np.linspace(origin_x, final_x, n_x)
    linspace_y = np.linspace(origin_y, final_y, n_y)
    # reshape private variable _var to shape [:, 1]
    if len(var.shape) == 1:
        _var = np.reshape(_var, (var.shape[0], 1))
    grid_x, grid_y = np.meshgrid(linspace_x, linspace_y)
    

