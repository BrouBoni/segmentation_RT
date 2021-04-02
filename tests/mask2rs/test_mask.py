from unittest import TestCase

import torchio as tio

from mask2rs.mask import Mask

TEST_CT_PATH = 'tests/data/cheese_png/ct'
TEST_MASK = 'tests/data/cheese_png/max.nii'


class TestMask(TestCase):
    def setUp(self):
        self.mask = Mask(TEST_MASK, ct_path=TEST_CT_PATH)

    def test_get_dicom_value(self):
        self.assertEqual('cheese', self.mask.get_dicom_value('PatientName'), 12)

    def test_coord_calculation(self):
        px, py, pz = self.mask.coordinate_mapping(224.0, 246.0, [-202.603515625, -452.603515625, 147])
        self.assertEqual([px, py, pz], [16.146484375, -212.369140625, 147.0])

    def test_coordinate(self):
        mask = tio.LabelMap(TEST_MASK)
        self.assertEqual(85, len(self.mask.coordinates(mask.data[0])))
