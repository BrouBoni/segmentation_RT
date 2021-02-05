from unittest import TestCase

from rs2mask.mask import Mask

TEST_IPP = 'rs2mask/data/314159'


class TestMask(TestCase):
    def setUp(self):
        self.mask = Mask(TEST_IPP)

    def test_get_dicom_value(self):
        self.assertEqual('PatientName', self.mask.get_dicom_value('PatientName'), 12)

    def test_coord_calculation(self):
        px, py, pz = self.mask.coord_calculation(224.0, 246.0, [-202.603515625, -452.603515625, 147])
        self.assertEqual([px, py, pz], [-24.978515625, -257.533203125, 147.0])

    def test_coordinate(self):
        self.assertEqual(62, len(self.mask.coordinates('heart')))
