from unittest import TestCase

import torchio as tio

from segmentation_rt.mask2rs.mask import Mask

TEST_CT_PATH = 'tests/test_data/cheese_png/ct'
TEST_MASK = 'tests/test_data/cheese_png/max.nii'


class TestMask(TestCase):
    def setUp(self):
        self.mask = Mask(TEST_MASK, ct_path=TEST_CT_PATH)

    def test_coordinate(self):
        mask = tio.LabelMap(TEST_MASK)
        self.assertEqual(6, len(self.mask.coordinates(mask.data[0])))
