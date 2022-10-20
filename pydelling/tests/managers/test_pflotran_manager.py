from unittest import TestCase
from pydelling.managers import PflotranManager
from pydelling.utils.configuration_utils import test_data_path


class TestBaseManager(TestCase):
    def setUp(self) -> None:
        self.base_manager = PflotranManager(str(test_data_path() / 'test_manager.in'))

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








