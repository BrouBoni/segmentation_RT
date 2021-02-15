import os

import numpy as np
import pydicom
from PIL import Image
from natsort import natsorted
from pydicom import dcmread
from pydicom.tag import Tag
from skimage import morphology
from skimage.measure import find_contours


class Mask:
    """
    Class that hold any mask aligned with a reference CT

    :param string path:
        Root directory which includes a mask folder.

    :param List[pydicom.dataset.FileDataset] ds_cts:
        CT Dataset.
    """

    def __init__(self, path, ds_cts=None):
        self.path = path
        self.masks_path = os.path.join(path, "masks")
        self.masks = os.listdir(self.masks_path)
        self.n_masks = len(self.masks)
        # ToDo verify reverse
        self.masks_files = natsorted([file for file in os.listdir(os.path.join(self.masks_path, self.masks[0]))
                                      if file.endswith("png")], reverse=True)

        self.ct_path = os.path.join(path, "ct")
        self.ct_files = natsorted([os.path.join(self.ct_path, ct) for ct in os.listdir(self.ct_path)
                                  if ct.endswith("dcm")])
        self.ds_ct = ds_cts or [dcmread(ct_file, force=True) for ct_file in self.ct_files]

        self.n_slices = len(self.masks_files)

        self.image_orientation_patient = list(self.get_dicom_value('ImageOrientationPatient', 0))
        self.pixel_spacing = list(self.get_dicom_value('PixelSpacing', 0))

    def __str__(self):
        message = f"Structure(s): {', '.join(self.masks)}"
        return message

    def get_dicom_value(self, tag, n_slice=0):
        """ Return a dicom tag value.

        :param tag: Dicom tag in keyword format
        :type tag: string
        :param n_slice: Slice number
        :type n_slice: int
        :return: the actual value. A regular value like a number or string (or list of them), or a Sequence.
        :rtype: :class:`pydicom.dataelem.DataElement`
        """
        return self.ds_ct[n_slice][Tag(tag)].value

    def coordinate_mapping(self, c, r, image_position_patient):
        """The mapping of pixel location (c,r) to Reference Coordinate System (RCS).

        :param c: column
        :type c: int
        :param r: row
        :type r: int
        :param image_position_patient:The x, y, and z coordinates of the upper left hand corner of the image, in mm.
        :type image_position_patient: :class:`pydicom.dataelem.DataElement`
        :return: The coordinates of the voxel (c,r) in the frame's image plane in units of mm.
        :rtype: list[int]
        """
        sx, sy, sz = np.array(image_position_patient, dtype=np.float32)
        delta_r, delta_c = np.array(list(self.pixel_spacing), dtype=np.float32)

        xx, xy, xz = np.array(self.image_orientation_patient[:3], dtype=np.float32)
        yx, yy, yz = np.array(self.image_orientation_patient[3:], dtype=np.float32)

        t_1 = np.array(list(self.get_dicom_value('ImagePositionPatient', 0))[:])
        t_n = np.array(list(self.get_dicom_value('ImagePositionPatient', self.n_slices - 1))[:])

        f = np.array([[yx, xx],
                      [yy, xy],
                      [yz, xz]])
        f11 = f[0, 0]
        f21 = f[1, 0]
        f31 = f[2, 0]
        f12 = f[0, 1]
        f22 = f[1, 1]
        f32 = f[2, 1]

        k = np.divide(np.subtract(t_n, t_1), (self.n_slices - 1))

        m1 = np.array([
            [f11 * delta_r, f12 * delta_c, k[0], sx],
            [f21 * delta_r, f22 * delta_c, k[1], sy],
            [f31 * delta_r, f32 * delta_c, k[2], sz],
            [0, 0, 0, 1]])

        m2 = np.array([r, c, 0, 1])

        px, py, pz, _ = np.asarray(np.dot(m1, m2))

        return px, py, pz

    def coordinates(self, mask_name):
        """Give the coordinates of the corresponding mask in theRCS and the SOP Instance UID`
         associated for each slice.

        :param mask_name: name of the mask.
        :type mask_name: str
        :return: List of ROI contour sequence.
        :rtype: list[(str,list[int])]
        """
        referenced_contour_data = []
        self.ct_files.reverse()
        for index, png in enumerate(self.masks_files):

            img_obj = Image.open(os.path.join(self.masks_path, mask_name, png)).convert('I')
            img_1d = np.array(list(img_obj.getdata()), bool)
            img_3d = np.reshape(img_1d, (img_obj.height, img_obj.width))
            # removing holes using a large value to be sure all the holes are removed
            img = morphology.remove_small_holes(img_3d, 100000, in_place=True)

            contours = find_contours(img)
            for contour in contours:
                if len(contour):
                    image_position_patient = self.get_dicom_value('ImagePositionPatient', index)
                    mask_coordinates = []
                    for coord in contour:
                        r, c = coord
                        x, y, z = self.coordinate_mapping(c, r, image_position_patient)
                        mask_coordinates.append(round(x, 4))
                        mask_coordinates.append(round(y, 4))
                        mask_coordinates.append(round(z, 4))
                    referenced_contour_data.append((self.ds_ct[index].SOPInstanceUID, mask_coordinates))

        return referenced_contour_data
