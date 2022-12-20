import unittest
from pydelling.readers import RasterFileReader
from pydelling.utils import test_data_path
import numpy.testing as npt


class TestSMeshMeshReader(unittest.TestCase):
    def setUp(self) -> None:
        self.top_surface = RasterFileReader(test_data_path() / 'top_surface.asc')
        self.bottom_surface = RasterFileReader(test_data_path() / 'bottom_surface.asc')
        self.bottom_surface_irregular = RasterFileReader(test_data_path() / 'bottom_surface_irregular.asc')
    def test_read_data(self):
        self.assertEqual(self.top_surface.nx, 250)
        self.assertEqual(self.top_surface.ny, 250)

    def test_write_read_irregular(self):
        self.bottom_surface_irregular.to_asc('test.asc')
        import matplotlib.pyplot as plt
        new_raster = RasterFileReader('test.asc')
        self.assertEqual(self.bottom_surface_irregular.nx, new_raster.nx)
        self.assertEqual(self.bottom_surface_irregular.ny, new_raster.ny)
        # Check the data is the same
        npt.assert_array_equal(self.bottom_surface_irregular.data, new_raster.data)




    def test_operations(self):
        # constant sum
        sum = self.top_surface + 1
        self.assertEqual(self.top_surface.values[0, 2] + 1, sum.values[0, 2])
        # Constant subtraction
        diff = self.top_surface - 1
        self.assertEqual(self.top_surface.values[:, 2].mean() - 1, diff.values[:, 2].mean())
        # Constant multiplication
        prod = self.top_surface * 2
        self.assertEqual(self.top_surface.values[:, 2].mean() * 2, prod.values[:, 2].mean())
        # Constant division
        quot = self.top_surface / 2
        self.assertEqual(self.top_surface.values[:, 2].mean() / 2, quot.values[:, 2].mean())
        # Raster sum
        sum = self.top_surface + self.bottom_surface
        self.assertEqual(self.top_surface.values[0, 2] + self.bottom_surface.values[0, 2], sum.values[0, 2])
        # Raster subtraction
        diff = self.top_surface - self.bottom_surface
        self.assertEqual(self.top_surface.values[0, 2] - self.bottom_surface.values[0, 2], diff.values[0, 2])
        # Raster multiplication
        prod = self.top_surface * self.bottom_surface
        self.assertEqual(self.top_surface.values[0, 2] * self.bottom_surface.values[0, 2], prod.values[0, 2])
        # Raster division
        quot = self.top_surface / self.bottom_surface
        self.assertEqual(self.top_surface.values[0, 2] / self.bottom_surface.values[0, 2], quot.values[0, 2])

    def test_get_data_from_coordinates(self):
        data = self.top_surface.get_data_from_coordinates(0.001240760000, 0.199278)
        self.assertAlmostEqual(data, 0.1182)

    def test_downsample(self):
        self.top_surface.downsample_data(2)
        self.assertEqual(self.top_surface.nx, 125)
        self.assertEqual(self.top_surface.ny, 125)

    def tearDown(self) -> None:
        from pathlib import Path
        if Path('test.asc').exists():
            Path('test.asc').unlink()



if __name__ == '__main__':
    unittest.main()