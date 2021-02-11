from unittest import TestCase

from mask2rs.mask import Mask

TEST_IPP = 'tests/data/cheese_png'


class TestMask(TestCase):
    def setUp(self):
        self.mask = Mask(TEST_IPP)

    def test_get_dicom_value(self):
        self.assertEqual('cheese', self.mask.get_dicom_value('PatientName'), 12)

    def test_coord_calculation(self):
        px, py, pz = self.mask.coord_calculation(224.0, 246.0, [-202.603515625, -452.603515625, 147])
        self.assertEqual([px, py, pz], [16.146484375, -212.369140625, 147.0])

    def test_coordinate(self):
        self.assertEqual(75, len(self.mask.coordinates('max')))
