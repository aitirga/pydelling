
"""
Base interface for a reader class
"""
import numpy as np
import logging
from pydelling.readers import BaseReader
logger = logging.getLogger(__name__)
from pydelling.config import config
import logging
from pathlib import Path
import h5py
import natsort
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
from collections import OrderedDict


class PflotranReader(BaseReader):
    def __init__(self, filename=None):
        self.filename = Path(filename) if filename else Path(config.pflotran_reader.filename)
        logger.info(f"Reading PFLOTRAN results file from {self.filename}")
        super().__init__(filename=self.filename)
        self.results = {}
        for time in self.time_keys:
            self.results[time] = PflotranResults(time=time, data=self.data[self.time_keys[time]])
        self.variables = list(self.results[self.time_values[0]].variable_keys)

    def open_file(self, filename):
        self.data: h5py.File = h5py.File(self.filename, 'r')
        # Obtain time keys
        self.time_dict_keys = natsort.natsorted(OrderedDict({float(key.split(' ')[2]): key for key in self.data.keys() if 'Time' in key}))
        self.time_keys = OrderedDict({float(key.split(' ')[2]): key for key in self.data.keys() if 'Time' in key})
        self.time_keys = {key: self.time_keys[key] for key in self.time_dict_keys}

    @property
    def time_values(self):
        return natsort.natsorted(self.time_keys.keys())

    @property
    def coordinates(self):
        temp_coordinates = self.data['Coordinates']
        temp_df = {'x[m]': np.array(temp_coordinates['X [m]']),
                                'y[m]': np.array(temp_coordinates['Y [m]']),
                                'z[m]': np.array(temp_coordinates['Z [m]']),
                                }
        return temp_df

    def get_data(self) -> np.ndarray:
        """
        Outputs the read data
        :return:
        """
        return np.array(0)

    def build_info(self):
        """
        Generates a dictionary containing the basic info of the read data
        :return:
        """
        self.info = {}

    def get_observation_point(self, variable, point=(0, 0, 0)):
        """
        Gets the time dependent evolution of a given variable at a given mesh point
        Args:
            variable: variable name to extract the data from
            point: mesh location

        Returns:
            Array containing the temporal evolution of the variable
        """
        temp_array = []
        for time in self.time_values:
            temp_array.append(self.results[time].results[variable][point[0], point[1], point[2]])
        pd_array = pd.DataFrame(np.array(temp_array), columns=[variable])
        return pd_array

    @property
    def mineral_names(self):
        temp_keys = [key.split('_')[0] for key in self.variables if 'VF' in key]
        return temp_keys

    @property
    def species_names(self):
        temp_keys = [key.split('_')[1] for key in self.variables if 'Total' in key]
        return temp_keys

    @property
    def x_centroid(self):
        return pd.DataFrame(np.diff(self.coordinates['x[m]']) + self.coordinates['x[m]'][0:-1], columns=['x[m]'])

    @property
    def y_centroid(self):
        return np.diff(self.coordinates['y[m]']) + self.coordinates['y[m]'][0:-1]

    @property
    def z_centroid(self):
        return np.diff(self.coordinates['z[m]']) + self.coordinates['z[m]'][0:-1]


    def get_mineral_vf_key(self, mineral):
        """
        Returns the correct key of the mineral volume fraction name
        Args:
            mineral: mineral name

        Returns:
            mineral volume fraction key
        """
        return f"{mineral}_VF [m^3 mnrl_m^3 bulk]"

    def get_mineral_rate_key(self, mineral):
        """
        Returns the correct key of the mineral rate name
        Args:
            mineral: mineral name

        Returns:
            mineral rate key
        """
        return f"{mineral}_Rate [mol_m^3_sec]"

    def get_mineral_si_key(self, mineral):
        """
        Returns the correct key of the mineral si name
        Args:
            mineral: mineral name

        Returns:
            mineral si key
        """
        return f"{mineral}_SI"

    def get_primary_species_key(self, species):
        """
        Returns the correct key of the given species name
        Args:
            species: specie name

        Returns:
            specie key
        """
        return f"Total_{species}"

    def plot_primary_species(self, type='1D', postprocess_dir='./postprocess'):
        """This method plots the primary species of a 1D model"""
        logger.info(f'Plotting [{self.species_names}] primary species from {self.filename}')
        postprocess_dir = Path(postprocess_dir)
        if type == '1D':
            for species in self.species_names:
                # Generate rate plots
                primary_species_key = self.get_primary_species_key (species)
                plot_df = pd.DataFrame()
                plt.clf()
                for time in config.postprocessing.times:
                    specie_data = self.results[time].results[primary_species_key][:, 0, 0]
                    specie_data_pd = pd.DataFrame (specie_data, columns=[f'{time} years'])
                    plot_df = plot_df.combine_first (specie_data_pd)
                plot_df = plot_df.combine_first (self.x_centroid)
                plot_df = plot_df.set_index ('x[m]')
                plot_df = plot_df.reindex(natsort.natsorted(plot_df.columns), axis=1)
                line_plot: plt.Axes = sns.lineplot (data=plot_df)
                line_plot.set_xlabel ('X [m]')
                line_plot.set_ylabel (f'{species}')
                plt.savefig(postprocess_dir / f'{species}.png')

    def plot_vf_variation(self, type='1D', postprocess_dir='./postprocess', ignored_minerals=[]):
        """This method plots the vf variation due to mineral precipitation"""
        logger.info(f'Plotting [{self.mineral_names}] primary species from {self.filename}')
        postprocess_dir = Path(postprocess_dir)
        postprocess_dir.mkdir(exist_ok=True)
        if type == '1D':
            for mineral in self.mineral_names:
                if mineral in ignored_minerals:
                    continue
                # Generate porosity variation plots
                primary_minerals_key = self.get_mineral_vf_key(mineral)
                plot_df = pd.DataFrame()
                plt.clf()
                # Save initial mineral volumes and porosity
                assert 'Porosity' in self.results[0.0].results, 'Porosity must be within the PFLOTRAN output variables'
                initial_porosity = self.results[0.0].results['Porosity']
                initial_mineral_vf = self.results[0.0].results[primary_minerals_key][:, 0, 0]
                for time in config.postprocessing.times:
                    mineral_vf = self.results[time].results[primary_minerals_key][:, 0, 0]
                    mineral_variation = mineral_vf - initial_mineral_vf
                    mineral_vf_pd = pd.DataFrame (mineral_variation, columns=[f'{time} years'])
                    plot_df = plot_df.combine_first (mineral_vf_pd)
                plot_df = plot_df.combine_first (self.x_centroid)
                plot_df = plot_df.set_index ('x[m]')
                plot_df = plot_df.reindex(natsort.natsorted(plot_df.columns), axis=1)
                line_plot: plt.Axes = sns.lineplot (data=plot_df)
                line_plot.set_xlabel ('X [m]')
                line_plot.set_ylabel (f'{mineral} volume fraction variation')
                plt.savefig(postprocess_dir / f'{mineral}.png')

    def plot_total_porosity_variation(self, type='1D', postprocess_dir='./postprocess', ignored_minerals=[]):
        """This method plots the vf variation due to mineral precipitation"""
        logger.info(f'Plotting [{self.mineral_names}] primary species from {self.filename}')
        postprocess_dir = Path(postprocess_dir)
        postprocess_dir.mkdir(exist_ok=True)
        plot_df = pd.DataFrame()

        if type == '1D':
            assert 'Porosity' in self.results[0.0].results, 'Porosity must be within the PFLOTRAN output variables'
            initial_porosity = self.results[0.0].results['Porosity'][:, 0, 0]
            for time in config.postprocessing.times:
                total_porosity_variation = np.zeros_like(initial_porosity)
                for mineral in self.mineral_names:
                    if mineral in ignored_minerals:
                        continue
                    # Generate porosity variation plots
                    primary_minerals_key = self.get_mineral_vf_key(mineral)
                    initial_mineral_vf = self.results[0.0].results[primary_minerals_key][:, 0, 0]
                    mineral_vf = self.results[time].results[primary_minerals_key][:, 0, 0]
                    mineral_variation = mineral_vf - initial_mineral_vf
                    total_porosity_variation += mineral_variation

                total_porosity_variation = total_porosity_variation / initial_porosity * 100
                mineral_vf_pd = pd.DataFrame(total_porosity_variation, columns=[f'{time} years'])
                plot_df = plot_df.combine_first(mineral_vf_pd)
            plot_df = plot_df.combine_first(self.x_centroid)
            plot_df = plot_df.set_index('x[m]')
            plot_df = plot_df.reindex(natsort.natsorted(plot_df.columns), axis=1)
            line_plot: plt.Axes = sns.lineplot (data=plot_df)
            line_plot.set_xlabel ('X [m]')
            line_plot.set_ylabel (f'{mineral} volume fraction variation')
        plt.savefig(postprocess_dir / f'{mineral}.png')

class PflotranResults:
    """
    This class stores the PFLOTRAN results
    """
    def __init__(self, time, data):
        """
        Args:
            time: timestep
            data: results dataset
        """
        self.time = time
        self.raw_data = data
        self.results = {key: np.array(self.raw_data[key]) for key in self.raw_data}
        self.variable_keys = self.results.keys()

    def __repr__(self):
        return f"Results of time {self.time}"

    @property
    def mineral_names(self):
        temp_keys = [key.split('_')[0] for key in self.variable_keys if 'VF' in key]
        return temp_keys

    @property
    def species_names(self):
        temp_keys = [key.split('_')[1] for key in self.variable_keys if 'Total' in key]
        return temp_keys





