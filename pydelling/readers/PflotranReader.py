
"""
Base interface for a reader class
"""
import logging

import numpy as np
import pandas

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
from collections import OrderedDict
from tqdm import tqdm
from pydelling.readers import PflotranProcessingUtils
import colorsys
import seaborn as sns



logger = logging.getLogger(__name__)

class PflotranReader(BaseReader, PflotranProcessingUtils):
    def __init__(self,
                 filename=None,
                 variables=None,
                 ):
        self.filename = Path(filename) if filename else Path(config.pflotran_reader.filename)
        logger.info(f"Reading PFLOTRAN results file from {self.filename}")
        super().__init__(filename=self.filename)
        self.results = {}
        if variables:
            self.variables = variables
        else:
            self.variables = list(self.data[self.time_keys[self.time_values[0]]])
        if 'Material_ID' not in self.variables:
            self.variables.append('Material_ID')

        for time in tqdm(self.time_keys, desc='Reading PFLOTRAN results'):
            data_slice = self.data[self.time_keys[time]]
            data_slice = {key: data_slice[key] for key in self.variables}
            self.results[time] = PflotranResults(time=time, data=data_slice)
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


    def get_variable_by_time_index(self, variable: str, time_index: int) -> np.ndarray:
        """
        Returns the results of a given variable at a given time index
        Args:
            variable: variable name
            time_index: time index

        Returns:
            results of the variable at the given time index
        """
        return self.results[self.time_values[time_index]].results[variable]

    def get_variable_by_time(self, variable: str, time: float) -> np.ndarray:
        """
        Returns the results of a given variable at a given time
        Args:
            variable: variable name
            time: time

        Returns:
            results of the variable at the given time
        """
        return self.results[time].results[variable]

    def get_results_by_time(self, time: float) -> dict:
        """
        Returns the results of a given time
        Args:
            time: time

        Returns:
            results of the given time
        """
        return self.results[time].results

    def get_results_by_time_index(self, time_index: int) -> dict:
        """
        Returns the results of a given time index
        Args:
            time_index: time index

        Returns:
            results of the given time index
        """
        return self.results[self.time_values[time_index]].results


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

    def plot_1D_slice_of_variable(self, variable,
                                  times=None,
                                  axis='x',
                                  coordinate=0,
                                  postprocess_dir='./postprocess',
                                  color='b',
                                  ) -> plt.Axes:
        """This method plots the variation of a variable along a given axis and coordinate"""
        fig, ax = plt.subplots()
        ax: plt.Axes
        ax_color = ax._get_lines.get_next_color()
        if times is None:
            times = [self.time_values[0]]
        elif times == 'all':
            times = self.time_values
        else:
            times = times

        for time_id, time in enumerate(times):
            data = self.get_variable_by_time(variable, time)
            data_slice = self.get_slice_from_coordinates(data=data,
                                                         axis=axis,
                                                         coordinate=coordinate
                                                         )
            dims = self.get_shape_dimensions(data_slice)
            n_times = len(times)
            # Convert hex to rgb
            next_color = tuple(int(ax_color.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4))
            original_color = colorsys.rgb_to_hls(*next_color)
            darker_color = colorsys.hls_to_rgb(original_color[0], 0.25 + 0.5 * time_id / n_times, original_color[2])
            data_slice = data_slice.flatten()
            x_data = self.axis_centroids(dims)
            ax.plot(x_data, data_slice, label=f'{time} years', color=darker_color)
            ax.set_xlabel(self.axis_translator[dims])
        ax.set_ylabel(variable)
        ax.grid()
        return ax

    def plot_1D_rigge_variable(self,
                               variable,
                                 times=None,
                                 axis='x',
                                 coordinate=0,
                                 postprocess_dir='./postprocess',
                                 color='b',
                                 ) -> plt.Axes:

        if times is None:
            times = [self.time_values[0]]
        elif times is 'all':
            times = self.time_values
        else:
            times = times

        df = pd.DataFrame()

        for time_id, time in enumerate(times):
            data = self.get_variable_by_time(variable, time)
            data_slice = self.get_slice_from_coordinates(data=data,
                                                         axis=axis,
                                                         coordinate=coordinate
                                                         )
            dims = self.get_shape_dimensions(data_slice)
            n_times = len(times)
            # Convert hex to rgb
            label_name = self.get_shape_dimensions(data_slice)
            data_slice = data_slice.flatten()
            x_data = self.axis_centroids(dims)
            times_var = [time] * len(x_data)
            df = pd.concat([df, pd.DataFrame({'x': x_data, 'y': data_slice, 'times': times_var})])


        a = 2
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(len(times), rot=-.25, light=.7)
        g = sns.FacetGrid(df, row="times", hue="times", aspect=15, height=.5, palette=pal)

        g.map(sns.lineplot, "x", 'y',
              clip_on=False,
            alpha=1, linewidth=1.5)
        # Fill the space between the line and the curve
        g.map(plt.fill_between, "x", "y", alpha=.2, clip_on=False)

        # Add a horizontal line to show the maximum value
        max_value = df['y'].max()
        # g.map(plt.axhline, y=max_value, lw=0.5, clip_on=True, color='k')
        def label(x, color, label):
            ax = plt.gca()
            ax.text(-.04, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)


        g.map(label, 'x')
        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=0.5)
        # Change the x axis labels

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        g.set_xlabels('x [m]')
        # Set y label in the middle
        g.fig.text(0.01, 0.5, variable, va='center', rotation='vertical')
        # Shrink the plot to fit the legend
        g.fig.subplots_adjust(left=0.1, bottom=0.15)








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





