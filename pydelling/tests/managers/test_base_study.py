from unittest import TestCase
from pydelling.managers import BaseStudy
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





