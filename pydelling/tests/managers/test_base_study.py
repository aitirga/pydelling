from unittest import TestCase
from pydelling.managers import BaseStudy, BaseCallback, BaseManager
from pydelling.utils.configuration_utils import test_data_path


class TestBaseStudy(TestCase):
    def setUp(self) -> None:
        self.base_manager = BaseStudy(str(test_data_path() / 'test_manager.in'))

    def test_replace_tag(self):
        regions = self.base_manager._find_tags('region')
        self.base_manager._replace_line(regions[0], ['REGION', 'pepe'])
        self.assertEqual(self.base_manager._get_line(regions[0]), 'REGION pepe')

    def test_to_file(self):
        self.base_manager.to_file(output_folder='test_folder')
        self.assertTrue(True)

    def test_callback(self):
        BaseManager.__abstractmethods__ = set()
        BaseCallback.__abstractmethods__ = set()

        dummy_study = BaseStudy(str(test_data_path() / 'test_manager.in'))
        dummy_study.add_callback(BaseCallback, kind='pre')

        dummy_manager = BaseManager()
        dummy_manager.add_study(dummy_study)
        dummy_manager.run()
        self.assertEqual(dummy_study.callbacks[0].kind, 'pre')
        self.assertEqual(dummy_study.callbacks[0].study, dummy_study)
        self.assertEqual(dummy_study.callbacks[0].manager, dummy_manager)
        self.assertEqual(dummy_study.callbacks[0].is_run, True)






