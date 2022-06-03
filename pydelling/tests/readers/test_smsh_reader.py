import unittest
from pydelling.readers import SMeshReader
from pydelling.utils import test_data_path


class TestSMeshMeshReader(unittest.TestCase):
    def test_read_data(self):
        fem_reader = SMeshReader(test_data_path() / "smesh_reader_data.smesh")
        self.assertEqual(fem_reader.n_nodes, 1373)
        self.assertEqual(fem_reader.n_elements, 3909)


if __name__ == '__main__':
    unittest.main()