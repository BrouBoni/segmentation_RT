import os
import shutil
from unittest import TestCase

import nibabel as nib

from rs2mask.dcm2mask import Dataset

TEST_IPP = 'tests/data/cheese_dcm'
TEST_RS = 'tests/data/cheese_dcm/cheese_dcm_1/RS1.2.752.243.1.1.20210208111802158.1580.88111.dcm'
TEST_NII = 'tests/data/ct.nii'


class TestDataset(TestCase):

    def setUp(self):
        structures = ['External', 'max', 'missing']
        root = TEST_IPP
        name = "dataset_cheese"
        self.dataset = Dataset(root, name, structures, '.')

    def tearDown(self):
        shutil.rmtree(self.dataset.path_dataset, ignore_errors=True)

    def test_get_rs(self):
        rs_paths = self.dataset.get_rs()
        self.assertEqual([TEST_RS], rs_paths)

    def test_find_structures(self):
        missing, not_missing = self.dataset.find_structures(0)
        self.assertEqual(len(missing), 1)
        self.assertEqual(['External', 'max'], list(not_missing))

    def test_nii_to_png(self):
        nii_object = nib.load(TEST_NII)
        self.dataset.nii_to_png('ct', nii_object, 'cheese_dcm_1')
        image_files = [f for f in os.listdir("tests/data/dataset_cheese/cheese_dcm_1/ct") if f.endswith('png')]
        self.assertEqual(len(image_files), nii_object.shape[2])

    def test_make_png(self):
        # ToDo
        self.fail()

    def test_sort_dataset(self):
        # ToDo
        self.fail()
