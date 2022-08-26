import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from pydelling.config import config
from pydelling.readers.iGPReader.io import BaseReader, iGPReader
from pydelling.readers.iGPReader.utils import utils

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt


class BoreholeReader(BaseReader):
    """This class reads borehole information from csv files and processes it to automatically assign to the mesh"""
    def __init__(self, igp_reader=None, filename=None):
        self.filename = filename if filename else config.data_files.borehole_data if config.data_files.borehole_data else None
        self.igp_reader: iGPReader = igp_reader
        self.boreholes_info = {}

    def run(self):
        self.read()
        self.set_materials_to_boreholes()
        if config.borehole_processing.permeability_from_borehole_data:
            self.set_permeability_from_excel()
        if config.borehole_processing.porosity_from_borehole_data:
            self.set_porosity_from_excel()
        if config.borehole_processing.make_material_plots:
            self.make_material_plots()

    def read(self):
        """Reads the borehole information data from a excel file"""
        if self.filename:
            if not Path(self.filename).suffix == ".xlsx" or Path(self.filename).suffix == ".xls":
                logger.error(f"{self.filename} is not an Excel file")
                raise AssertionError(f"{self.filename} is not an Excel file")
            if config.borehole_processing.anisotropic_borehole_data:
                logger.info('Processing the file as containing anisotropic permeability values')
                self.data = pd.read_excel(self.filename, header=3, sheet_name=None)
                for borehole in self.data:
                    self.data[borehole] = self.data[borehole].iloc[1:]
                    self.data[borehole] = self.data[borehole]
                    self.data[borehole].rename(columns={'Permeability (m^2)': 'kx',
                                                'Unnamed: 3': 'ky',
                                                'Unnamed: 4': 'kz',
                                                }, inplace=True)
                    self.data[borehole]['kmean'] = stats.gmean(np.array([self.data[borehole]['kx'],
                                                                         self.data[borehole]['ky'],
                                                                         self.data[borehole]['kz'],
                                                                         ]).astype(np.float)
                                                               )
            else:
                self.data = pd.read_excel(self.filename, header=3, sheet_name=None)
            _excel_file: Dict[str, pd.DataFrame] = pd.read_excel(self.filename, sheet_name=None)
        for borehole in _excel_file:
            self.boreholes_info[borehole] = {"x": _excel_file[borehole].iloc[0, 1],
                                        "y": _excel_file[borehole].iloc[0, 2],
                                        "z": _excel_file[borehole].iloc[0, 3],
                                             }

    def set_permeability_from_excel(self):
        logger.info(f"Setting the permeability values based on the {self.filename} file")
        for material in self.igp_reader.material_dict:
            # Compute an approximated permeability value for each material
            geometric_mean_boreholes = []
            for borehole in self.data:
                filtered_borehole = self.data[borehole][self.data[borehole]["Material_name"] == material]
                if config.borehole_processing.anisotropic_borehole_data:
                    geometric_mean_boreholes.append(stats.gmean(filtered_borehole["kmean"]))
                else:
                    geometric_mean_boreholes.append(stats.gmean(filtered_borehole["Permeability (m^2)"]))
            material_permeability = stats.gmean(geometric_mean_boreholes)
            self.igp_reader.material_info[material]["permeability"] = material_permeability

    def set_porosity_from_excel(self):
        logger.info(f"Setting the porosity values based on the {self.filename} file")
        for material in self.igp_reader.material_dict:
            # Compute an approximated permeability value for each material
            arithmetic_mean_boreholes = []
            for borehole in self.data:
                filtered_borehole = self.data[borehole][self.data[borehole]["Material_name"] == material]
                arithmetic_mean_boreholes.append(stats.hmean(filtered_borehole["Porosity (-)"]))
            material_porosity = stats.hmean(arithmetic_mean_boreholes)
            self.igp_reader.material_info[material]["porosity"] = material_porosity

    def set_materials_to_boreholes(self):
        """This method should assign a material ID to each depth of the borehole"""
        logger.info("Setting materials to boreholes")
        for borehole in self.boreholes_info:
            self.data[borehole].insert(self.data[borehole].shape[1], "Material_ID", np.zeros(self.data[borehole].shape[0]))
            # create a coordinate dataframe
            _borehole_coordinates = self.data[borehole][["Z (mAOD)", "Material_ID"]]
            _borehole_coordinates.insert(0, "x", self.boreholes_info[borehole]["x"] * np.ones(self.data[borehole].shape[0]))
            _borehole_coordinates.insert(1, "y", self.boreholes_info[borehole]["y"] * np.ones(self.data[borehole].shape[0]))
            _borehole_coordinates.columns = ["x", "y", "z", "Material_ID"]
            logger.debug(f"Calculating material IDs for borehole {borehole}")
            _material_id_vector = utils.get_material_id_from_centroids(igp_reader=self.igp_reader,
                                                                       borehole_coordinates=_borehole_coordinates,
                                                                       )
            self.data[borehole]["Material_ID"] = _material_id_vector
            self.data[borehole]["Material_name"] = [self.igp_reader.material_names[material_id] for material_id in self.data[borehole]["Material_ID"]]

    def make_material_plots(self):
        # print(self.data['Borehole1Data'])
        if config.borehole_processing.permeability_from_borehole_data:
            if config.borehole_processing.anisotropic_borehole_data:
                self.borehole_plot(variable='kx')
                self.borehole_plot(variable='ky')
                self.borehole_plot(variable='kz')
                self.borehole_plot(variable='kmean')
            else:
                self.borehole_plot(variable='Permeability (m^2)')
        if config.borehole_processing.porosity_from_borehole_data:
            self.borehole_plot(variable='Porosity (-)')

    def borehole_plot(self, variable) -> [plt.Figure, plt.Axes]:
        """This method plots the permeability and porosity values taken from the boreholes for each material"""
        logger.info(f'Plotting borehole {variable} data vs "Z (mAOD)"')
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots()
        if config.borehole_processing.make_material_plots.grid:
            ax.grid(zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel('Z (mAOD)')
        plot_data = {}
        dictionary_built = False
        for borehole in self.data:
            # Build data structure
            for material in self.data[borehole]['Material_name'].unique():
                if dictionary_built == False:
                    plot_data[material] = {'x': [], 'y': []}
            # Add plot data based on the boreholes
            dictionary_built = True
            for material in self.data[borehole]['Material_name'].unique():
                plot_data[material]['x'].append(self.data[borehole][self.data[borehole]['Material_name'] == material][variable].values.astype(np.float))
                if config.borehole_processing.make_material_plots.plot_cross_correlation:
                    cross_correlation_var = config.borehole_processing.make_material_plots.plot_cross_correlation.variable
                    plot_data[material]['y'].append(self.data[borehole][self.data[borehole]['Material_name'] == material][cross_correlation_var].values.astype(np.float))
                    logger.info(f'Plotting cross-correlation plots using {cross_correlation_var}')
                    ax.semilogy()
                    ax.set_ylabel(config.borehole_processing.make_material_plots.plot_cross_correlation.label if config.borehole_processing.make_material_plots.plot_cross_correlation.label else f'{cross_correlation_var} [m^2]')


                else:
                    plot_data[material]['y'].append(self.data[borehole][self.data[borehole]['Material_name'] == material]['Z (mAOD)'].values.astype(np.float))

        for material in plot_data:
            plot_x_values = np.concatenate(plot_data[material]['x']).astype(np.float)
            plot_y_values = np.concatenate(plot_data[material]['y']).astype(np.float)
            ax.scatter(x=plot_x_values,
                       y=plot_y_values,
                       label=material,
                       )
            ax.semilogx()
            if config.borehole_processing.make_material_plots.perm_range:
                plot_range = np.array(config.borehole_processing.make_material_plots.perm_range).astype(np.float)
                if variable != 'Porosity (-)':
                    ax.set_xlim(left=plot_range[0],
                                right=plot_range[1],
                                )
        ax.legend()
        if variable is 'Porosity (-)':
            ax.set_xlabel('Porosity [-]')
        elif variable is 'kx':
            ax.set_xlabel('Permeability-x $[m^2]$')
        elif variable is 'ky':
            ax.set_xlabel('Permeability-y $[m^2]$')
        elif variable is 'kz':
            ax.set_xlabel('Permeability-z $[m^2]$')
        elif variable is 'kmean':
            ax.set_xlabel('Geometric mean of permeability $[m^2]$')
        else:
            ax.set_xlabel(f"{variable}")
        plt.savefig(utils.get_output_path() / f'{variable.replace(" ", "_")}_plot.png')
        return fig, ax
