import os
import shutil
from unittest import TestCase

from segmentation_rt.rs2mask.rs2mask import Dataset

TEST_IPP = 'tests/test_data/cheese_dcm'
TEST_EXPORT = 'tests/test_data/dataset_cheese'
TEST_RS = 'tests/test_data/cheese_dcm/cheese_dcm_1/RS1.2.752.243.1.1.20210208111802158.1580.88111.dcm'


class TestDataset(TestCase):

    def setUp(self):
        structures = ['External', 'max', 'missing']
        self.dataset = Dataset(TEST_IPP, TEST_EXPORT, structures)

    def test_name(self):
        name = str(self.dataset)
        self.assertEqual(name, 'dataset_cheese')

    def tearDown(self):
        shutil.rmtree(self.dataset.export_path, ignore_errors=True)

    def test_get_rs(self):
        rs_paths = self.dataset.get_rs()
        self.assertEqual([TEST_RS], rs_paths)

    def test_find_structures(self):
        missing, not_missing = self.dataset.find_structures(0)
        self.assertEqual(len(missing), 1)
        self.assertEqual(['External', 'max'], list(not_missing))

    def test_make(self):
        self.dataset.make()
        self.assertTrue(os.path.exists(os.path.join(TEST_EXPORT, "cheese_dcm_1", "ct.nii")))
