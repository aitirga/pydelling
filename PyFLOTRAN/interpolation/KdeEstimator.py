import pandas as pd
from sklearn.neighbors import KernelDensity
from PyFLOTRAN.utils.decorators import set_run
import numpy as np
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__file__)


class KdeEstimator:
    """
    This class performs a KDE estimation on a given dataset and provides useful methods
    to plot and manipulate the estimated distributions.
    """
    is_run: bool
    kde_estimator: KernelDensity
    def __init__(self, data:pd.DataFrame=None, kernel='gaussian', bandwidth=1000):
        self.data = data
        self.kernel = kernel
        self.bandwidth = bandwidth
        if type(self.data) == pd.Series:
            self.data = pd.DataFrame(self.data)

    @set_run
    def fit(self, data: pd.DataFrame=None):
        """
        Fit the provided dataset to the corresponding kernel distribution
        """
        if data:
            self.data = data
        self.kde_estimator: KernelDensity = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(self._training_data)
        assert self.kde_estimator, 'Provide some data to work with'
        return self.kde_estimator

    def plot_1D(self, variable=None, n=100, savefig=None):
        """
        Plots the trained KDE using the original data bounds
        Returns: Axes object of the generated plot
        """
        if not variable:
            variable = self.data.columns[0]

        sampling_data = np.arange(self.data[variable].min(), self.data[variable].max(), n)
        predicted_distribution = self.kde_estimator.score_samples(sampling_data.reshape(-1, 1))
        predicted_distribution = np.exp(predicted_distribution)

        # Plot result
        fig, ax = plt.subplots()
        fig: plt.Figure
        ax: plt.Axes
        ax.hist(self.data[variable].values, bins=30,
                density=True,
                alpha=0.5,
                label=variable,
                )
        ax.plot(sampling_data,
                predicted_distribution,
                alpha=0.80,
                label=f'{variable}-predicted KDE'
                )
        plt.legend()
        plt.grid()
        ax.set_ylabel('Density [-]')
        ax.set_xlabel('Integration time [y]')
        if savefig:
            plt.savefig(savefig)
        else:
            plt.show()
        return ax

    @property
    def _training_data(self):
        assert not self.data.empty, 'Provide some data to work with'
        return self.data.values.reshape(-1, self.data.shape[1])

    def sample(self, *args, **kwargs):
        return self.kde_estimator.sample(*args, **kwargs)