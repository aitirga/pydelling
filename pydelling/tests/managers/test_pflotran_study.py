from unittest import TestCase
from pydelling.managers import PflotranStudy
from pydelling.utils.configuration_utils import test_data_path


class TestPflotranCase(TestCase):
    def setUp(self) -> None:
        self.pflotran_study = PflotranStudy(str(test_data_path() / 'test_manager.in'))

    def test_get_regions(self):
        regions = self.pflotran_study.get_regions()
        self.assertEqual(regions, ['all', 'inlet', 'sides', 'top_below', 'top_above', 'bottom', 'initial', 'Sand'])

    def test_get_time(self):
        time = self.pflotran_study.get_simulation_time(time_unit='d')
        self.assertEqual(time, 365.0)
        new_time = 2
        new_time_unit = 'd'
        self.pflotran_study.replace_simulation_time(new_time=new_time, time_unit=new_time_unit)
        time = self.pflotran_study.get_simulation_time(time_unit=new_time_unit)
        self.assertEqual(time, new_time)

    def test_get_region_file(self):
        region = 'all'
        file = self.pflotran_study.get_region_file(region)
        self.assertEqual(file, None)
        region = 'top_below'
        file = self.pflotran_study.get_region_file(region)
        self.assertEqual(file, './input_files/top_below.ex')

    def test_replace_region_file(self):
        region = 'top_below'
        new_file = './input_files/test_replace.ex'
        self.pflotran_study.replace_region_file(region, new_file)
        file = self.pflotran_study.get_region_file(region)
        self.assertEqual(file, new_file)

    def test_checkpoint(self):
        times = [1, 2, 3]
        time_unit = 'd'
        self.pflotran_study.add_checkpoint(times=times, time_unit=time_unit)
        checkpoints = self.pflotran_study.get_checkpoint()
        self.assertEqual(checkpoints, 'TIMES d 1.0 2.0 3.0')
        manager_no_checkpoint = PflotranStudy(str(test_data_path() / 'test_pflotran_manager_nocheckpoint.in'))
        self.assertEqual(manager_no_checkpoint.get_checkpoint(), None)
        manager_no_checkpoint.add_checkpoint(times=times, time_unit=time_unit)
        self.assertEqual(manager_no_checkpoint.get_checkpoint(), 'TIMES d 1.0 2.0 3.0')
        manager_no_checkpoint.to_file()

    def test_datasets(self):
        datasets = ['dirichletpressure', 'topflow']
        self.assertEqual(self.pflotran_study.get_datasets(), datasets)
        self.pflotran_study.add_dataset(name='dirichletpressure', filename='pepe.dat', hdf5_dataset_name='pepe')
        self.pflotran_study.add_dataset(name='pepe', filename='pepe.dat', hdf5_dataset_name='pepe')
        new_datasets = ['dirichletpressure', 'topflow', 'pepe']
        self.assertListEqual(sorted(self.pflotran_study.get_datasets()), sorted(new_datasets))














