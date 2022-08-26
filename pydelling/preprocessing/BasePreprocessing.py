import pandas as pd

from pydelling.utils.decorators import set_run


class BasePreprocessing:
    def __init__(self, data: pd.DataFrame = None, filename=None):
        if data is not None:
            self.data = data
        elif filename:
            self.data = pd.read_csv(filename)

    @set_run
    def run(self):
        """
        This method should be implemented for each preprocessing class. It should run the preprocessing algorithm.
        Returns: the result of the preprocessing algorithm
        """
        return 1