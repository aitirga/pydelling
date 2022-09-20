"""
Base interface for a reader class
"""
import logging

import numpy as np

from pydelling.readers import BaseReader

logger = logging.getLogger(__name__)
from pydelling.config import config
import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


class PflotranObservationPointReader(BaseReader):
    observation_point: np.ndarray
    observation_boundary: str
    observation_node: int
    variables: dict
    def __init__(self, filename=None):
        self.filename = Path(filename) if filename else Path(config.pflotran_reader.filename)
        logger.info(f"Reading PFLOTRAN observation point results file from {self.filename}")
        super().__init__(filename=self.filename)
        self.results = self.data.copy()

        # self.results = {}
        # for time in self.time_keys:
        #     self.results[time] = PflotranResults(time=time, data=self.data[self.time_keys[time]])
        # self.variables = list(self.results[self.time_values[0]].variable_keys)

    def open_file(self, filename):
        self.data: pd.DataFrame = pd.read_csv(self.filename,
                                              skiprows=1,
                                              header=None,
                                              delim_whitespace=True
                                              )
        header = pd.read_csv(self.filename,
                             nrows=0
                             ).columns.tolist()
        self.data.columns = header

        # Rename time column
        time_column: str = self.data.columns.to_list()[0]
        # self.data.rename(columns={time_column: time_column.strip().strip('"')}, inplace=True)
        # # Get observation point information
        # temp_column_name: str = self.data.columns[1]
        # temp_column_name = temp_column_name.replace('(', '').replace(')', '')
        # splitted_column_name = temp_column_name.split(' ')
        # self.observation_point = np.array([splitted_column_name[-3],
        #                                   splitted_column_name[-2],
        #                                   splitted_column_name[-3]],
        #                                   ).astype(float)
        # self.observation_node = int(splitted_column_name[-4])
        # self.observation_boundary = splitted_column_name[-5]
        # Get output column names
        # self.variables = {}
        # for column in self.data.columns:
        #     if 'Time' in column:
        #         continue
        #     test_column_name: str = column
        #     test_column_name_split = test_column_name.split('-')[1].split()
        #     variable_str = []
        #     flag_list = ['east', 'west', 'north', 'south', 'top', 'bottom']
        #     for piece in test_column_name_split:
        #         if piece in flag_list:
        #             break
        #         else:
        #             variable_str.append(piece)
        #     variable_str = ' '.join(variable_str)
        #     self.variables[variable_str] = column
        #     self.data.rename(columns={column: variable_str}, inplace=True)

    @property
    def mineral_names(self):
        temp_keys = [key for key in self.variables if 'VF' in key]
        return temp_keys

    @property
    def total_species_names(self):
        temp_keys = [key for key in self.variables if 'Total' in key]
        return temp_keys

    @property
    def free_species_names(self):
        temp_keys = [key for key in self.variables if 'Free' in key]
        return temp_keys

    def plot_variable(self, variable,
                      delete_previous=True,
                      label=None ) -> plt.Axes:
        logger.info(f'Creating lineplot of {variable}')
        if delete_previous:
            plt.clf()
        lineplot: plt.Axes = plt.plot(self.time_series,
                                self.results[variable])[0]
        lineplot.set_label(f'{variable if label is None else label}')
        return lineplot

    def to_csv(self, filename='postprocess/results.csv', variables=None) -> pd.DataFrame:
        self.create_postprocess_dict()
        logger.info(f'Exporting results to csv')
        if variables:
            plot_results: pd.DataFrame = self.results[variables]
            plot_results.to_csv(filename)
        else:
            self.results.to_csv(filename, index=False)

        # print(self.variables)

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

    @property
    def time_series(self):
        return self.results.iloc[:, 0]

    @property
    def columns(self):
        return self.results.columns

