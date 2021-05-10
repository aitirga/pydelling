import pandas as pd


class BasePreprocessing:
    def __init__(self, data: pd.DataFrame):
        self.data = data