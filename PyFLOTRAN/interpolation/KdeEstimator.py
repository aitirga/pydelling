import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats.kde import gaussian_kde
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
    def __init__(self, data:pd.DataFrame=None, kernel='gaussian', bandwidth=1000, package='scikit'):
        self.data = data
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.package = package
        if type(self.data) == pd.Series:
            self.data = pd.DataFrame(self.data)

    @set_run
    def fit(self, data: pd.DataFrame=None):
        """
        Fit the provided dataset to the corresponding kernel distribution
        """
        if data:
            self.data = data
        logger.info(f'Fitting KDE generator for {self.data.columns.to_list()} variables')
        self.kde_estimator: KernelDensity = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(self._training_data)
        return self.kde_estimator

    def plot_1d_comparison_histograms(self, variable=None, n=10000, savefig=None, bins=30):
        plt.clf()
        fig, ax = plt.subplots()
        fig: plt.Figure
        ax: plt.Axes
        if not variable:
            variable = self.data.columns[0]

        sampled_values = pd.DataFrame(self.kde_estimator.sample(n), columns=self.data.columns)

        ax.hist(self.data[variable].values, bins=bins,
                density=True,
                alpha=0.5,
                label=variable,
                )

        ax.hist(sampled_values[variable].values, bins=bins,
                density=True,
                alpha=0.5,
                label=f"{variable}-kde",
                )

        plt.legend()
        if savefig:
            plt.savefig(savefig)
        else:
            plt.show()
        return ax

    def plot_1d(self, variable=None, n=100, savefig=None):
        """
        Plots the trained KDE using the original data bounds
        Returns: Axes object of the generated plot
        """
        plt.clf()
        if not variable:
            variable = self.data.columns[0]
        logger.info(f'Generating 1D comparison plot for {variable} variable')

        temp = [np.linspace(self.data[column].min(), self.data[column].max(), n) for column in self.data]
        sampling_data = np.meshgrid(*temp)
        sampling_data: np.ndarray = np.vstack(map(np.ravel, sampling_data)).T.reshape(-1, self.data.shape[1])

        predicted_distribution = self.kde_estimator.score_samples(sampling_data)
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