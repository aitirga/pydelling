import numpy as np

from .Fracture import Fracture
from typing import List
import pandas as pd
import logging
from tqdm import tqdm
import plotly.graph_objects as go
from tabulate import tabulate


logger = logging.getLogger(__name__)


class DfnPreprocessor(object):
    dfn: List[Fracture] = []

    def __init__(self):
        pass

    def load_fractures(self, pd_df: pd,
                       dip='dip',
                       dip_dir='dip-direction',
                       x='position-x',
                       y='position-y',
                       z='position-z',
                       size='size',
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
                size=row[size]
            )

    def add_fracture(self, dip, dip_dir, x, y, z, size):
        """Add individual fracture to the dfn object.
        """
        self.dfn.append(Fracture(
            dip=dip,
            dip_dir=dip_dir,
            x=x,
            y=y,
            z=z,
            size=size
        ))

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

    def visualize_dfn(self, add_centroid=True, fracture_color='blue', size_color=False):
        """Visualizes the dfn object."""
        self.fig = self.generate_dfn_plotly(add_centroid=add_centroid, fracture_color=fracture_color, size_color=size_color)
        self.fig.show()

    def export_dfn_image(self, filename='dfn.png', add_centroid=True, fracture_color='blue', *args, **kwargs, ):
        logger.info(f'Exporting dfn image to {filename}')
        self.fig = self.generate_dfn_plotly(add_centroid=add_centroid, fracture_color=fracture_color)
        self.fig.write_image(filename, *args, **kwargs)

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



