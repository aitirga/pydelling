import unittest
from PyFLOTRAN.readers import CentroidReader
from PyFLOTRAN.config import config
from PyFLOTRAN.utils import test_data_path
import logging
# logging.getLogger().setLevel("CRITICAL")
from pathlib import Path


class CentroidReaderCase(unittest.TestCase):
    """
    This class tests the iGPReader[PyFLOTRAN.readers.iGPReader.io.iGPReader.iGPReader] class implementation
    """
    def setUp(self) -> None:
        self.centroid_reader = CentroidReader(test_data_path() / "centroid_reader_data.csv", header=True, split_key=",")

    def test_centroid_data(self):
        self.assertEqual(self.centroid_reader.get_data()[0][0], 0.0)

    def test_write_data(self):
        write_path = Path().cwd() / "test.csv"
        self.centroid_reader.dump_to_csv(write_path)
        write_path.unlink()


if __name__ == '__main__':
    unittest.main()
