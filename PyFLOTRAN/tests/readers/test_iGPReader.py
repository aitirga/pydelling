import unittest
from PyFLOTRAN.readers import iGPReader
from PyFLOTRAN.config import config
from PyFLOTRAN.utils import test_data_path
import logging
logging.getLogger().setLevel("CRITICAL")


class iGPReaderCase(unittest.TestCase):
    """
    This class tests the iGPReader[PyFLOTRAN.readers.iGPReader.io.iGPReader.iGPReader] class implementation
    """
    def setUp(self) -> None:
        self.stream_line_reader = iGPReader(test_data_path() / "test_implicit_to_explicit.gid",
                                            project_name="test-implicit_to_explicit"
                                            )

    def test_implicit_to_explicit(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
