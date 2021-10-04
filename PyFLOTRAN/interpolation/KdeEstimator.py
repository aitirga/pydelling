from sklearn.neighbors import KernelDensity
from PyFLOTRAN.utils.decorators import set_run


class KdeEstimator:
    """
    This class performs a KDE estimation on a given dataset and provides useful methods
    to plot and manipulate the estimated distributions.
    """
    is_run: bool
    def __init__(self, data, method='gaussian'):
        self.data = data
        self.method = method

    @set_run
    def fit(self):
        """
        Fit the provided dataset to the corresponding kernel distribution
        """

    def plot(self, n=100):
        """
        Plots the trained KDE using the original data bounds
        Returns: Axes object of the generated plot
        """
        assert self.is_run, 'Train the KDE before plotting the results'
