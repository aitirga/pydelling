from .Fracture import Fracture
from typing import List
import pandas as pd
import logging
from tqdm import tqdm

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



