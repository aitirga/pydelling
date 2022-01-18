import numpy as np

from .Fracture import Fracture
from typing import List
import pandas as pd
import logging
from tqdm import tqdm
import plotly.graph_objects as go


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
        pass

    def visualize_dfn(self, add_centroid=True):
        """Visualizes the dfn object."""
        self.generate_dfn_plotly(add_centroid=add_centroid)

    def export_dfn_image(self, filename):
        pass

    def generate_dfn_plotly(self, add_centroid=True):
        ''' Generates a plotly figure of the dfn object.

        Returns: A plotly figure

        '''
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
                        color='rgb(0, 0, 0)',
                        symbol='circle',
                        opacity=0.65
                    )
                ))

            fig.add_trace(go.Mesh3d(
                x=fracture_sides[:, 0],
                y=fracture_sides[:, 1],
                z=fracture_sides[:, 2],
                color='#1f77b4',
                opacity=0.75
            ))
        fig.show()





