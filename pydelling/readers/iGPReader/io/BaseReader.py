from typing import Dict

import pandas as pd
from copy import deepcopy


class BaseReader:
    """Base class for the different reader subclasses"""
    filename: str
    data: Dict[str, pd.DataFrame]

    def read(self):
        """This function should read the data automatically from the config file"""
        pass

    def copy(self):
        """This function returns a copy of the reader"""
        return deepcopy(self)
