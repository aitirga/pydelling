import unittest
from pydelling.readers.FemReader import FemReader
from pydelling.utils import test_data_path


class TestFemReader(unittest.TestCase):
    def test_read_data(self):
        fem_reader = FemReader(test_data_path() / "fem_reader_data.fem")
        self.assertEqual(fem_reader.n_nodes, 1373)
        self.assertEqual(fem_reader.n_elements, 3909)


if __name__ == '__main__':
    unittest.main()