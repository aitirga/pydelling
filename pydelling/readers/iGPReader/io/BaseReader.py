from typing import Dict

import pandas as pd


class BaseReader:
    """Base class for the different reader subclasses"""
    filename: str
    data: Dict[str, pd.DataFrame]

    def read(self):
        """This function should read the data automatically from the config file"""
        pass
