import unittest
from pydelling.readers import SmeshReader
from pydelling.utils import test_data_path


class TestSMeshMeshReader(unittest.TestCase):
    def test_read_data(self):
        smesh_reader = SmeshReader(test_data_path() / 'smesh_test_data.smesh')

        self.assertEqual(smesh_reader.n_nodes, 4)
        self.assertEqual(smesh_reader.n_elements, 1)




if __name__ == '__main__':
    unittest.main()