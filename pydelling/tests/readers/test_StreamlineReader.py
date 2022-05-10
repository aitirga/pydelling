import unittest
from pydelling.readers import StreamlineReader
from pydelling.config import config
from pydelling.utils import test_data_path
from pathlib import Path


class StreamlineReaderCase(unittest.TestCase):
    def setUp(self) -> None:
        self.stream_line_reader = StreamlineReader(filename=test_data_path() / config.streamline_reader.file)

    def test_read_case(self):
        self.assertEqual(self.stream_line_reader.raw_data["Points:0"].loc[0], 0.066709)

    def test_generate_streams(self):
        self.assertEqual(self.stream_line_reader.stream_data.get_group(0).iloc[1, 0], 1.0)

    def test_compute_arrival_times(self):
        arrival_times = self.stream_line_reader.compute_arrival_times()
        self.assertEqual(arrival_times[10], 14114.0)

    def test_write_csv(self):
        write_path = Path.cwd() / "test_streamlines.csv"
        self.stream_line_reader.dump_to_csv(write_path)
        write_path.unlink()



if __name__ == '__main__':
    unittest.main()
