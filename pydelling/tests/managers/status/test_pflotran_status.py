import unittest
from pydelling.managers.status import PflotranStatus
from pydelling.utils import test_data_path


class TestPflotranStatus(unittest.TestCase):
    def setUp(self) -> None:
        self.pflotran_status = PflotranStatus(test_data_path() / 'pflotran_status.out')

    def test_class(self):
        self.assertEqual(self.pflotran_status.wall_clock_time, 27361.0)
        self.assertTrue(self.pflotran_status.is_done)
        self.assertEqual(len(self.pflotran_status.times), len(self.pflotran_status.dts))
        self.assertIsNone(self.pflotran_status.progress)
        self.pflotran_status.add_total_time(11000.0)
        self.assertEqual(self.pflotran_status.progress, 1.0)


if __name__ == '__main__':
    unittest.main()


