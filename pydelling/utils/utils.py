"""
Contains general purpose utility functions
"""
import os
from pathlib import Path

import numpy as np
import yaml

import pydelling.interpolation as interpolation
import pydelling.readers as readers

try:
    from pydelling.paraview_processor.filters import BaseFilter, PlotOverLineFilter
except:
    from pydelling.paraview_processor.filters import BaseFilter
import pandas as pd
from box import Box


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


def aperture_from_a_xy_point_old(dataset: BaseFilter, x_point, y_point, line_interpolator, variable=None, target_value=1.0, threshold=0.45, line_resolution=100):
    """
    This method computes the aperture at a given position in the XY plane.
    Args:
        x_point: X coordinate
        y_point: Y coordinate
        variable: The variable to consider that indicates the aperture
        target_value: Value of the variable when the fracture is open
        threshold: Value of the maximum variation from the target_value |line[variable] - target_value| < threshold is assumed
        line_resolution: Resolution of the line interpolation that is being used
    Returns:
        The value of the aperture at a given point
    """
    point_1 = [x_point, y_point, dataset.z_min]
    point_2 = [x_point, y_point, dataset.z_max]
    line_interpolator.set_points(point_1=point_1, point_2=point_2)
    line_interpolator.set_line_resolution(line_resolution)

    line_interpolation_point_data = line_interpolator.point_data
    dataset_mesh_points = dataset.mesh_points
    if variable:
        # If a target variable is defined,
        # print(abs(line_interpolation_point_data[variable] - target_value))
        line_interpolation = line_interpolation_point_data[abs(line_interpolation_point_data[variable] - target_value) < threshold]
        z_points: pd.Series = dataset_mesh_points.iloc[line_interpolation.index]["z"]
        if len(z_points) > 0:
            # aperture = abs(z_points.max() - z_points.min())
            aperture = z_points.max()
        else:
            aperture = 0.0
        return aperture

    if not variable:
        # Calculate directly the aperture
        z_points: pd.Series = dataset_mesh_points.iloc[line_interpolation_point_data.index]["z"]
        if len(z_points) > 0:
            aperture = abs(z_points.max() - z_points.min())
        else:
            aperture = 0.0
        return aperture

def aperture_from_a_xy_point(dataset: BaseFilter, x_point, y_point, line_interpolator, variable=None, target_value=1.0, threshold=0.45, line_resolution=100, method='explicit'):
    """
    This method computes the aperture at a given position in the XY plane.
    Args:
        x_point: X coordinate
        y_point: Y coordinate
        variable: The variable to consider that indicates the aperture
        target_value: Value of the variable when the fracture is open
        threshold: Value of the maximum variation from the target_value |line[variable] - target_value| < threshold is assumed
        line_resolution: Resolution of the line interpolation that is being used
        method: Method to use to calculate the aperture. It can be 'explicit' or 'implicit'
    Returns:
        The value of the aperture at a given point
    """
    point_1 = [x_point, y_point, dataset.z_min]
    point_2 = [x_point, y_point, dataset.z_max]
    line_interpolator.set_points(point_1=point_1, point_2=point_2)
    line_interpolator.set_line_resolution(line_resolution)
    line_interpolator_resolution = (dataset.z_max - dataset.z_min) / line_resolution


    line_interpolation_point_data = line_interpolator.point_data
    dataset_mesh_points = dataset.mesh_points
    if variable:
        # If a target variable is defined,
        # print(abs(line_interpolation_point_data[variable] - target_value))
        # line_interpolation = line_interpolation_point_data[abs(line_interpolation_point_data[variable] - target_value) < threshold]
        # z_points: pd.Series = dataset_mesh_points.iloc[line_interpolation.index]["z"]
        if method == 'explicit':
            line_interpolation = line_interpolation_point_data[abs(line_interpolation_point_data[variable] - target_value) < threshold]
            z_points: pd.Series = dataset_mesh_points.iloc[line_interpolation.index]["z"]
            aperture = len(z_points) * line_interpolator_resolution
            return aperture
        elif method == 'implicit':
            eps_field = (line_interpolation_point_data[variable] - line_interpolation_point_data[variable].min())
            eps_field = eps_field / eps_field.max()
            aperture = (eps_field * line_interpolator_resolution).sum()
            return aperture

    if not variable:
        # Calculate directly the aperture
        z_points: pd.Series = dataset_mesh_points.iloc[line_interpolation_point_data.index]["z"]
        if len(z_points) > 0:
            aperture = abs(z_points.max() - z_points.min())
        else:
            aperture = 0.0
        return aperture


def get_root_path() -> Path:
    """Returns path to the root of the project"""
    return Path(__file__).parent


def get_config_path() -> Path:
    """Returns path to the root of the project"""
    return Path(__file__).parent / "config"


def test_data_path() -> Path:
    """Returns path to the root of the project"""
    return Path(__file__).parent.parent / "tests/test_data"

def runtime_path():
    return Path(os.getcwd())


def read_local_config():
    def read_config(config_file: Path = "./config.yaml"):
        """
        Reads the configuration file
        :param config_file:
        :return:
        """
        with open(config_file) as file:
            context = yaml.load(file, Loader=yaml.FullLoader)
        return Box(context, default_box=True)

    _config_file = list(
        Path(os.getcwd()).glob("**/*config.yml") and Path(os.getcwd()).glob("**/*config.yaml") and Path().cwd().glob(
            "*config*.yml") and Path().cwd().glob(
            "*config*.yaml"))
    _config_file = _config_file if _config_file else list(Path(__file__).parent.glob("config.yml"))
    assert len(_config_file) == 1, "Please provide a configuration file that has a '*config.yaml' name structure"
    config = read_config(config_file=_config_file[0])
    return config


def sample_values_from_dict(input_dict: dict, n: int, write_to_file=True, return_generator=False) -> list:
    """
    This method reads a dictionary and generates n samples from them based on the
    following criteria.

    The structure of the dictionary should be as follows:

    name_of_dict:
        material_name_i:
            type: type of the desired distribution
            option_keys: these keys depend on the chosen distribution type
        material_name_i:

    Type of implemented distributions:
        constant[value]: sets a constant value for the variable
        log_normal[mean, std]: generates a log-normal distribution based on the mean, std values
        specified in natural scale.

    Args:
        write_to_file: optional argument that writes a csv file with the generated cases. Defaults to True.
        d: dictionary to extract the samples from
        n: integer specifying the number of generated samples
    Returns:
        A dictionary containing the sampled results
    """
    class BaseDistribution:
        def __init__(self):
            pass
        def run(self) -> float:
            """
            This method generates a value of the given distribution
            Returns: a float number containing the value
            """

    class ConstantDistribution(BaseDistribution):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def run(self):
            return self.value

    class NormalDistribution(BaseDistribution):
        def __init__(self, mean, std, log=False):
            super().__init__()
            self.mean = float(mean)
            self.std = float(std)
            self.log = log

        def run(self):
            if not self.log:
                return np.random.normal(self.mean, self.std)
            else:
                return np.random.lognormal(self.mean, self.std)

    # Process each material and create sample generators
    generator_dict = {}
    for material in input_dict:
        material_dict = input_dict[material]
        if material_dict['type'] == 'constant':
            generator_dict[material] = ConstantDistribution(value=material_dict['value'])
        elif material_dict['type'] == 'normal':
            generator_dict[material] = NormalDistribution(mean=material_dict['mean'],
                                                          std=material_dict['std'],
                                                          )
        elif material_dict['type'] == 'log-normal' or material_dict['type'] == 'log_normal':
            generator_dict[material] = NormalDistribution(mean=material_dict['mean'],
                                                          std=material_dict['std'],
                                                          log=True,
                                                          )
        elif material_dict['type'] == 'log_normal-two_values':
            y_min_log = np.log(float(material_dict['y_min']))
            y_max_log = np.log(float(material_dict['y_max']))
            mean = (y_min_log + y_max_log) / 2.0
            std = (mean - y_min_log) / 2.0

            generator_dict[material] = NormalDistribution(mean=mean,
                                                          std=std,
                                                          log=True,
                                                          )

    final_list = []
    for sample_id in range(n):
        current_case = {}
        for generator in generator_dict:
            current_case[generator] = generator_dict[generator].run()
        final_list.append(current_case)
    if write_to_file:
        df_test = pd.DataFrame(final_list)
        df_test.to_csv('SA_cases.csv', index=False)
    if return_generator:
        return generator_dict
    return final_list


