import unittest
from PyFLOTRAN.readers import StreamlineReader
from PyFLOTRAN.config import config
from PyFLOTRAN.utils import test_data_path
import logging
logging.getLogger().setLevel("CRITICAL")


class StreamlineReaderCase(unittest.TestCase):
    def setUp(self) -> None:
        self.stream_line_reader = StreamlineReader(filename=test_data_path() / config.streamline_reader.file)

    def test_read_case(self):
        self.assertEqual(self.stream_line_reader.raw_data["Points:0"].loc[0], 0.066709)

    def test_generate_streams(self):
        self.assertEqual(self.stream_line_reader.stream_data.get_group(0).iloc[1, 0], 1.0)



if __name__ == '__main__':
    unittest.main()
