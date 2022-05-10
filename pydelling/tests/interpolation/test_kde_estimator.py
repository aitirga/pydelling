import unittest
from pydelling.interpolation import KdeEstimator
import numpy as np
import pandas as pd


class KdeEstimatorCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data = pd.DataFrame(np.random.normal(size=1000))

    def test_set_up(self):
        self.kde_estimator = KdeEstimator()

    def test_train_gaussian(self):
        self.kde_estimator = KdeEstimator(data=self.test_data, bandwidth=0.0001)
        self.kde_estimator.run()
        test_samples = self.kde_estimator.sample(100)
        self.assertLess(abs(test_samples.mean()), 0.5)

if __name__ == '__main__':
    unittest.main()