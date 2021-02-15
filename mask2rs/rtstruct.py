import datetime
import os
import random

from natsort import natsorted
from pydicom import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, UID

from mask2rs.mask import Mask

COMMON_TAGS = {'PatientSex', 'SeriesInstanceUID', 'SeriesNumber', 'SeriesDate', 'AccessionNumber',
               'PatientID', 'SpecificCharacterSet', 'SeriesDescription', 'StudyDate', 'StudyDescription',
               'PatientName', 'StudyTime', 'InstanceNumber', 'StudyInstanceUID', 'ReferringPhysicianName',
               'PositionReferenceIndicator', 'PatientBirthDate', 'ManufacturerModelName', 'SeriesTime',
               'FrameOfReferenceUID', 'StudyID', 'Modality'}


class RTStruct:

    def __init__(self, path):
        self.path = path
        self.ct_path = os.path.join(path, "ct")
        self.ct_files = natsorted([os.path.join(self.ct_path, ct) for ct in os.listdir(self.ct_path)
                                   if ct.endswith("dcm")])

        self.ds_cts = [dcmread(ct_file) for ct_file in self.ct_files]
        self.ds_ct_sop_instance_uid = [ds_ct.SOPInstanceUID for ds_ct in self.ds_cts]

        self.ds_ct_reference = self.ds_cts[0]

        self.mask = Mask(self.path, self.ds_cts)
        self.ds_rs = Dataset()

        # RS meta
        self.ds_rs.file_meta = FileMetaDataset()
        self.ds_rs.file_meta.MediaStorageSOPClassUID = 'RT Structure Set Storage'
        self.ds_rs.file_meta.MediaStorageSOPInstanceUID = '1.2.345.678.9.1.11111111111111111.1111.11111'

        # RS transfer syntax
        self.ds_rs.is_little_endian = True
        self.ds_rs.is_implicit_VR = True

        # RS creation date/time
        date_time = datetime.datetime.now()
        self.ds_rs.ContentDate = date_time.strftime('%Y%m%d')

        time_str = date_time.strftime('%H%M%S')
        self.ds_rs.ContentTime = time_str

        # RS common tags
        for tag in COMMON_TAGS:
            if tag in self.ds_ct_reference:
                self.ds_rs[tag] = self.ds_ct_reference[tag]

        # RS series
        self.ds_rs.Modality = 'RTSTRUCT'
        self.ds_rs.SOPClassUID = UID('1.2.840.10008.5.1.4.1.1.481.3')
        self.ds_rs.SOPInstanceUID = generate_uid()
        self.ds_rs.ManufacturerModelName = "RSpy"
        self.ds_rs.StructureSetLabel = 'RS: Unapproved'
        self.ds_rs.ApprovalStatus = 'UNAPPROVED'

        # RS empty sequences
        self.ds_rs.add_new('RTROIObservationsSequence', 'SQ', Sequence())
        self.ds_rs.add_new('StructureSetROISequence', 'SQ', Sequence())
        self.ds_rs.add_new('ROIContourSequence', 'SQ', Sequence())
        self.ds_rs.add_new('ReferencedFrameOfReferenceSequence', 'SQ', Sequence())

        # RS referenced frame of reference sequence
        self.__add_referenced_frame_of_reference_sequence()

    def __str__(self):
        message = f"{self.ds_rs.Modality}"
        return message

    def __add_referenced_frame_of_reference_sequence(self):
        self.ds_rs.ReferencedFrameOfReferenceSequence.append(Dataset())
        referenced_frame_of_reference_sequence = self.ds_rs.ReferencedFrameOfReferenceSequence[-1]
        referenced_frame_of_reference_sequence.FrameOfReferenceUID = self.ds_rs.FrameOfReferenceUID

        referenced_frame_of_reference_sequence.add_new('RTReferencedStudySequence', 'SQ', Sequence())
        referenced_frame_of_reference_sequence.RTReferencedStudySequence.append(Dataset())
        rt_referenced_study_sequence = referenced_frame_of_reference_sequence.RTReferencedStudySequence[-1]
        rt_referenced_study_sequence.ReferencedSOPClassUID = UID('1.2.840.10008.3.1.2.3.1')
        rt_referenced_study_sequence.ReferencedSOPInstanceUID = self.ds_ct_reference.SeriesInstanceUID

        rt_referenced_study_sequence.add_new('RTReferencedSeriesSequence', 'SQ', Sequence())
        rt_referenced_study_sequence.RTReferencedSeriesSequence.append(Dataset())
        rt_referenced_series_sequence = rt_referenced_study_sequence.RTReferencedSeriesSequence[-1]
        rt_referenced_series_sequence.SeriesInstanceUID = self.ds_ct_reference.SeriesInstanceUID

        rt_referenced_series_sequence.add_new('ContourImageSequence', 'SQ', Sequence())

        for i, sop_instance_uid in enumerate(self.ds_ct_sop_instance_uid):
            rt_referenced_series_sequence.ContourImageSequence.append(Dataset())
            contour_image_sequence = rt_referenced_series_sequence.ContourImageSequence[i]
            contour_image_sequence.ReferencedSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
            contour_image_sequence.ReferencedSOPInstanceUID = sop_instance_uid

    def init_sequence(self):
        self.ds_rs.RTROIObservationsSequence.append(Dataset())
        self.ds_rs.StructureSetROISequence.append(Dataset())
        self.ds_rs.ROIContourSequence.append(Dataset())

    def add_roi_observation_sequence(self, structure, index):
        roi_observation_sequence = self.ds_rs.RTROIObservationsSequence[index]
        roi_observation_sequence.ObservationNumber = structure["ObservationNumber"]
        roi_observation_sequence.ReferencedROINumber = structure["ReferencedROINumber"]
        roi_observation_sequence.ROIObservationLabel = structure["ROIObservationLabel"]
        roi_observation_sequence.RTROIInterpretedType = structure["RTROIInterpretedType"]
        roi_observation_sequence.ROIInterpreter = structure["ROIInterpreter"]

    def add_structure_set_roi_sequence(self, structure, index):
        structure_set_roi_sequence = self.ds_rs.StructureSetROISequence[index]
        structure_set_roi_sequence.ROINumber = structure["ReferencedROINumber"]
        structure_set_roi_sequence.ReferencedFrameOfReferenceUID = self.ds_rs.FrameOfReferenceUID
        structure_set_roi_sequence.ROIName = structure["ROIObservationLabel"]
        structure_set_roi_sequence.ROIGenerationAlgorithm = structure["ROIGenerationAlgorithm"]

    def add_roi_contour_sequence(self, structure, index, coordinates):
        roi_contour_sequence = self.ds_rs.ROIContourSequence[index]
        roi_contour_sequence.ReferencedROINumber = structure["ReferencedROINumber"]
        roi_contour_sequence.ROIDisplayColor = random.sample(range(0, 255), 3)

        roi_contour_sequence.add_new('ContourSequence', 'SQ', Sequence())

        for i, (sop_instance_uid, coordinate) in enumerate(coordinates):
            roi_contour_sequence.ContourSequence.append(Dataset())
            contour_sequence = roi_contour_sequence.ContourSequence[-1]
            contour_sequence.ContourGeometricType = "CLOSED_PLANAR"
            contour_sequence.NumberOfContourPoints = int(len(coordinate) / 3)
            contour_sequence.ContourNumber = i
            contour_sequence.ContourData = coordinate

            contour_sequence.add_new('ContourImageSequence', 'SQ', Sequence())
            contour_sequence.ContourImageSequence.append(Dataset())
            contour_image_sequence = contour_sequence.ContourImageSequence[-1]
            contour_image_sequence.ReferencedSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
            contour_image_sequence.ReferencedSOPInstanceUID = sop_instance_uid

    def add_structure_to_dataset(self, structure, index, coordinate):
        self.init_sequence()
        self.add_roi_observation_sequence(structure, index)
        self.add_structure_set_roi_sequence(structure, index)
        self.add_roi_contour_sequence(structure, index, coordinate)

    def create(self):
        for index, mask_name in enumerate(self.mask.masks):
            structure = {"ObservationNumber": str(index),
                         "ReferencedROINumber": str(index),
                         "ROIObservationLabel": mask_name,
                         "RTROIInterpretedType": "ORGAN",
                         "ROIInterpreter": "",
                         "ROIGenerationAlgorithm": "AUTOMATIC"
                         }
            coordinates = self.mask.coordinates(mask_name)
            coordinates.reverse()
            self.add_structure_to_dataset(structure, index, coordinates)
            print(mask_name, " ", str(index + 1), "/", str(len(self.mask.masks)))

    def save(self, name='RS.dcm'):
        self.ds_rs.save_as(name, write_like_original=False)
