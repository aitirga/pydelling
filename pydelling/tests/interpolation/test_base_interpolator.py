import unittest
from pydelling.interpolation import BaseInterpolator
from pydelling.readers import iGPReader, CentroidReader
from pydelling.utils import test_data_path
import numpy


class BaseInterpolatorCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh_data = iGPReader(test_data_path() / "test_implicit_to_explicit.gid", project_name="test").get_mesh()
        self.interpolation_data = CentroidReader(test_data_path() / "centroid_reader_data.csv", separator=",", header=True).get_data()
        self.base_interpolator = BaseInterpolator(mesh_data=self.mesh_data, interpolation_data=self.interpolation_data)

    def test_set_up(self):
        numpy.testing.assert_array_equal(self.base_interpolator.mesh, self.mesh_data)
        numpy.testing.assert_array_equal(self.base_interpolator.data, self.interpolation_data)

    def test_interpolation(self):
        interpolated_data = self.base_interpolator.run()
        numpy.testing.assert_array_equal(interpolated_data, self.mesh_data[:, 0])


if __name__ == '__main__':
    unittest.main()
