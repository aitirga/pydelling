from unittest import TestCase
from pydelling.managers import PflotranCase
from pydelling.utils.configuration_utils import test_data_path


class TestPflotranCase(TestCase):
    def setUp(self) -> None:
        self.base_manager = PflotranCase(str(test_data_path() / 'test_manager.in'))

    def test_get_regions(self):
        regions = self.base_manager.get_regions()
        self.assertEqual(regions, ['all', 'inlet', 'sides', 'top_below', 'top_above', 'bottom', 'initial', 'Sand'])

    def test_get_time(self):
        time = self.base_manager.get_simulation_time(time_unit='d')
        self.assertEqual(time, 365.0)
        new_time = 2
        new_time_unit = 'd'
        self.base_manager.replace_simulation_time(new_time=new_time, time_unit=new_time_unit)
        time = self.base_manager.get_simulation_time(time_unit=new_time_unit)
        self.assertEqual(time, new_time)

    def test_get_region_file(self):
        region = 'all'
        file = self.base_manager.get_region_file(region)
        self.assertEqual(file, None)
        region = 'top_below'
        file = self.base_manager.get_region_file(region)
        self.assertEqual(file, './input_files/top_below.ex')

    def test_replace_region_file(self):
        region = 'top_below'
        new_file = './input_files/test_replace.ex'
        self.base_manager.replace_region_file(region, new_file)
        file = self.base_manager.get_region_file(region)
        self.assertEqual(file, new_file)

    def test_checkpoint(self):
        times = [1, 2, 3]
        time_unit = 'd'
        self.base_manager.add_checkpoint(times=times, time_unit=time_unit)
        checkpoints = self.base_manager.get_checkpoint()
        self.assertEqual(checkpoints, 'TIMES d 1.0 2.0 3.0')
        manager_no_checkpoint = PflotranCase(str(test_data_path() / 'test_pflotran_manager_nocheckpoint.in'))
        self.assertEqual(manager_no_checkpoint.get_checkpoint(), None)
        manager_no_checkpoint.add_checkpoint(times=times, time_unit=time_unit)
        self.assertEqual(manager_no_checkpoint.get_checkpoint(), 'TIMES d 1.0 2.0 3.0')
        manager_no_checkpoint.to_file()













