import os
from unittest import TestCase

from mask2rs.rtstruct import RTStruct

TEST_IPP = 'tests/data/cheese_png'


class TestRTStruct(TestCase):

    def setUp(self):
        self.rtstruct = RTStruct(TEST_IPP)
        self.structure = {"ObservationNumber": "0",
                          "ReferencedROINumber": "0",
                          "ROIObservationLabel": "max",
                          "RTROIInterpretedType": "ORGAN",
                          "ROIInterpreter": "",
                          "ROIGenerationAlgorithm": "AUTOMATIC"
                          }

    def test_init_sequence(self):
        roi_observation_sequence = len(self.rtstruct.ds_rs.RTROIObservationsSequence)
        structure_set_roi_sequence = len(self.rtstruct.ds_rs.StructureSetROISequence)
        roi_contour_sequence = len(self.rtstruct.ds_rs.ROIContourSequence)
        self.assertEqual([roi_observation_sequence, structure_set_roi_sequence, roi_contour_sequence], [0, 0, 0])

        self.rtstruct.init_sequence()
        roi_observation_sequence = len(self.rtstruct.ds_rs.RTROIObservationsSequence)
        structure_set_roi_sequence = len(self.rtstruct.ds_rs.StructureSetROISequence)
        roi_contour_sequence = len(self.rtstruct.ds_rs.ROIContourSequence)
        self.assertEqual([roi_observation_sequence, structure_set_roi_sequence, roi_contour_sequence], [1, 1, 1])

    def test_add_roi_observation_sequence(self):
        self.rtstruct.init_sequence()
        self.rtstruct.add_roi_observation_sequence(self.structure, 0)

        self.assertEqual(str(self.rtstruct.ds_rs.RTROIObservationsSequence[0].ObservationNumber), "0")
        self.assertEqual(str(self.rtstruct.ds_rs.RTROIObservationsSequence[0].ReferencedROINumber), "0")
        self.assertEqual(str(self.rtstruct.ds_rs.RTROIObservationsSequence[0].ROIObservationLabel), "max")
        self.assertEqual(str(self.rtstruct.ds_rs.RTROIObservationsSequence[0].RTROIInterpretedType), "ORGAN")
        self.assertEqual(str(self.rtstruct.ds_rs.RTROIObservationsSequence[0].ROIInterpreter), "")

    def test_add_structure_set_roi_sequence(self):
        self.rtstruct.init_sequence()
        self.rtstruct.add_structure_set_roi_sequence(self.structure, 0)

        self.assertEqual(str(self.rtstruct.ds_rs.StructureSetROISequence[0].ROINumber), "0")
        self.assertEqual(str(self.rtstruct.ds_rs.StructureSetROISequence[0].ReferencedFrameOfReferenceUID),
                         self.rtstruct.ds_rs.FrameOfReferenceUID)
        self.assertEqual(str(self.rtstruct.ds_rs.StructureSetROISequence[0].ROIName), "max")
        self.assertEqual(str(self.rtstruct.ds_rs.StructureSetROISequence[0].ROIGenerationAlgorithm), "AUTOMATIC")

    def test_add_roi_contour_sequence(self):
        self.rtstruct.init_sequence()
        coordinates = self.rtstruct.mask.coordinates("max")
        self.rtstruct.add_roi_contour_sequence(self.structure, 0, coordinates)

        self.assertEqual(str(self.rtstruct.ds_rs.ROIContourSequence[0].ReferencedROINumber), "0")
        self.assertEqual(len(self.rtstruct.ds_rs.ROIContourSequence[0].ContourSequence[0]), 5)

    def test_save(self):
        save_path = "RS.dcm"
        self.rtstruct.save()
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)
