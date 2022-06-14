import numpy as np

from .Fracture import Fracture
from .Fault import Fault
from typing import List
import pandas as pd
import logging
from tqdm import tqdm
import plotly.graph_objects as go
from tabulate import tabulate
from pathlib import Path


logger = logging.getLogger(__name__)


class DfnPreprocessor(object):
    dfn: List[Fracture] = []
    faults: List[Fault] = []

    def __getitem__(self, item):
        return self.dfn[item]

    def __init__(self):
        self.clean_dfn()

    def clean_dfn(self):
        """Removes dfn object"""
        self.dfn = []
        self.faults = []


    def load_fractures(self, pd_df: pd,
                       dip='dip',
                       dip_dir='dip-direction',
                       x='position-x',
                       y='position-y',
                       z='position-z',
                       size='size',
                       aperture=None,
                       aperture_constant=None,
                       transmissivity_constant=None,
                       storativity_constant=None
                       ):
        """
        Loads the fractures from a pandas dataframe to the dfn object.
        Args:
            pd_df: pandas dataframe containing the fractures

        Returns:

        """
        for index, row in tqdm(pd_df.iterrows()):
            self.add_fracture(
                dip=row[dip],
                dip_dir=row[dip_dir],
                x=row[x],
                y=row[y],
                z=row[z],
                size=row[size],
                aperture_constant=aperture_constant,
                aperture=aperture,
                transmissivity_constant=transmissivity_constant,
                storativity_constant=storativity_constant
            )

    def load_fractures_from_polygons_and_apertures(self,
                       polygons,
                       apertures=None,
                       radii=None,
                       aperture_constant=None,
                       transmissivity_constant=None,
                       storativity_constant=None):
        for idx, polygon in enumerate(polygons):
            self.add_fracture(
                polygon=polygon,
                aperture=None, #aperture=apertures[idx],
                size=radii[idx] * 2 if radii is not None else None,
                aperture_constant=aperture_constant,
                transmissivity_constant=transmissivity_constant,
                storativity_constant=storativity_constant,
            )


    def add_fracture(self,
                     x=None,
                     y=None,
                     z=None,
                     dip=None,
                     dip_dir=None,
                     size=None,
                     aperture=None,
                     aperture_constant=1E-3,
                     transmissivity_constant=None,
                     storativity_constant=None,
                     polygon=None,
                     ):
        """Add individual fracture to the dfn object.
        """
        self.dfn.append(Fracture(
            dip=dip,
            dip_dir=dip_dir,
            x=x,
            y=y,
            z=z,
            size=size,
            aperture=aperture,
            aperture_constant=aperture_constant,
            transmissivity_constant=transmissivity_constant,
            storativity_constant=storativity_constant,
            polygon=polygon,
        ))

    def add_fault(self, filename=None,
                  mesh=None,
                  aperture=None):
        """Adds a fault to the dfn object."""
        if aperture is None:
            logger.warning(f'No aperture specified for fault {filename}')
        if isinstance(filename, Fault):
            self.faults.append(filename)
        elif isinstance(filename, str) or isinstance(filename, Path):
            self.faults.append(Fault(filename=filename, mesh=mesh, aperture=aperture))
        else:
            logger.error('Fault filename must be a string or Fault object')
            raise TypeError('Fault filename must be a string or Fault object')

    def summary(self):
        """Prints a summary of the dfn object."""
        print(tabulate(
            [
                ['Number of fractures', len(self.dfn)],
                ['Max size', f"{self.max_size:1.2f} m"],
                ['Min size', f"{self.min_size:1.2f} m"]
            ],
            headers=['Parameter', 'Value'],
            tablefmt='grid',
            numalign='center',
        ))

    def __len__(self):
        return len(self.dfn)

    def visualize_dfn(self, add_centroid=True, fracture_color='blue', size_color=False):
        """Visualizes the dfn object."""
        self.fig = self.generate_dfn_plotly(add_centroid=add_centroid, fracture_color=fracture_color, size_color=size_color)
        self.fig.show()

    def export_dfn_image(self, filename='dfn.png', add_centroid=True, fracture_color='blue', *args, **kwargs, ):
        logger.info(f'Exporting dfn image to {filename}')
        self.fig = self.generate_dfn_plotly(add_centroid=add_centroid, fracture_color=fracture_color)
        self.fig.write_image(filename, *args, **kwargs)

    def to_obj(self, filename='dfn.obj', method='v1'):
        '''Exports the dfn object to stl format.'''
        logger.info(f'Exporting dfn object to {filename}')
        obj_file = open(filename, 'w')
        obj_file.write('# Created by pydelling\n')
        obj_file.write('o dfn\n')
        global_id = 1
        for fracture in tqdm(self.dfn):
            obj_file.write(fracture.to_obj(global_id=global_id, method=method))
            global_id += fracture.n_side_points
        for fault in self.faults:
            fault_obj = fault.to_obj(global_id=global_id)
            obj_file.write(fault_obj)
            global_id += fault.num_points


    def to_dfnworks(self, filename='dfn.dat', method='v1'):
        '''Exports the dfn object to dfnworks format.'''
        logger.info(f'Exporting dfn object to {filename}')
        dfn_file = open(filename, 'w')
        n_total_fractures = len(self.dfn)
        dfn_file.write(f'nPolygons: {n_total_fractures}\n')
        for fracture in tqdm(self.dfn):
            side_points = fracture.get_side_points(method=method)
            dfn_file.write(f'{len(side_points)} ')
            for point in side_points:
                dfn_file.write(f'{{{point[0]},{point[1]},{point[2]}}}')
            dfn_file.write('\n')

    def shift(self, x_shift=0, y_shift=0, z_shift=0):
        """Shifts the dfn object."""
        logger.info(f'Shifting dfn object by {x_shift}, {y_shift}, {z_shift}')
        for fracture in self.dfn:
            fracture.shift(x_shift, y_shift, z_shift)


    def generate_dfn_plotly(self, add_centroid=False, size_color=False, fracture_color='blue'):
        ''' Generates a plotly figure of the dfn object.

        Returns: A plotly figure

        '''
        logger.info('Generating plotly figure')
        fig = go.Figure()
        for fracture in tqdm(self.dfn):
            fracture_sides = fracture.get_side_points()
            if add_centroid:
                fig.add_trace(go.Scatter3d(
                    x=[fracture.x_centroid],
                    y=[fracture.y_centroid],
                    z=[fracture.z_centroid],
                    mode='markers',
                    marker=dict(
                        size=3.5,
                        color='black',
                        symbol='circle',
                        opacity=0.65
                    )
                ))
            # Add color depending on the fracture size
            if size_color:
                A = 255 / (self.max_size - self.min_size)
                B = 255 - A * self.max_size
                value = int(A * fracture.size + B)
                color = f'rgb({value}, 0, 0)'
            else:
                color = fracture_color
            fig.add_trace(go.Mesh3d(
                x=fracture_sides[:, 0],
                y=fracture_sides[:, 1],
                z=fracture_sides[:, 2],
                color=color,
                opacity=0.75,
            ))

        return fig

    @property
    def max_size(self):
        """Returns the maximum size of the dfn object."""
        return max([fracture.size for fracture in self.dfn])

    @property
    def min_size(self):
        """Returns the minimum size of the dfn object."""
        return min([fracture.size for fracture in self.dfn])

    def plot_radii_histogram(self, filename='radii_histogram.png'):
        """Plots the radii histogram."""
        logger.info(f'Plotting radii histogram to {filename}')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.hist([(fracture.size / 2) for fracture in self.dfn], bins=100)
        return fig, ax

    def plot_aperture_histogram(self, filename='aperture_histogram.png'):
        """Plots the aperture histogram."""
        logger.info(f'Plotting aperture histogram to {filename}')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.hist([fracture.aperture for fracture in self.dfn], bins=100)
        return fig, ax

    def plot_transmissivity_histogram(self, filename='transmissivity_histogram.png'):
        """Plots the transmissivity histogram."""
        logger.info(f'Plotting aperture histogram to {filename}')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.hist([fracture.transmissivity for fracture in self.dfn], bins=100)
        return fig, ax

    def plot_hkx_histogram(self, filename='hkx_histogram.png'):
        """Plots the x-hydraulic conductivity histogram."""
        logger.info(f'Plotting hk_x histogram to {filename}')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.hist([(fracture.transmissivity / fracture.aperture) for fracture in self.dfn], bins=100)
        return fig, ax


    def plot_storativity_histogram(self, filename='storativity_histogram.png'):
        """Plots the storativity histogram."""
        logger.info(f'Plotting aperture histogram to {filename}')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.hist([fracture.storativity for fracture in self.dfn], bins=100)
        return fig, ax

    @property
    def apertures(self):
        return [fracture.aperture for fracture in self.dfn]

    def __add__(self, other):
        """Adds two dfn objects."""
        if not isinstance(other, DfnPreprocessor):
            raise TypeError(f'{other} is not a DfnPreprocessor object')

        logger.info('Adding dfn objects')
        new_dfn = DfnPreprocessor()
        new_dfn.dfn = self.dfn + other.dfn
        new_dfn.faults = self.faults + other.faults
        return new_dfn

    @property
    def min_x(self):
        return min([fracture.x_centroid for fracture in self.dfn])

    @property
    def max_x(self):
        return max([fracture.x_centroid for fracture in self.dfn])

    @property
    def min_y(self):
        return min([fracture.y_centroid for fracture in self.dfn])

    @property
    def max_y(self):
        return max([fracture.y_centroid for fracture in self.dfn])

    @property
    def min_z(self):
        return min([fracture.z_centroid for fracture in self.dfn])

    @property
    def max_z(self):
        return max([fracture.z_centroid for fracture in self.dfn])




