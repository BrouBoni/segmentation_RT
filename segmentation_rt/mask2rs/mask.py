import os

import numpy as np
import torchio as tio
from natsort import natsorted
from pydicom import dcmread
from pydicom.tag import Tag
from scipy import interpolate
from skimage import morphology
from skimage.measure import find_contours


class Mask:
    """
    Class that hold any structures aligned with a reference CT.

    :param :class:`tio.LABEL` mask:
        Path to the mask or a tio.Label.

    :param string ct_path:
        Path to the CT folder.

    :param List[pydicom.dataset.FileDataset] ds_cts:
        CT Dataset.
    """

    def __init__(self, mask, ct_path=None, ds_cts=None):

        if ct_path is None and ds_cts is None:
            raise ValueError('At least ct_path should be provided')

        self.masks = mask if not isinstance(mask, str) else tio.LabelMap(mask)
        self.transform = tio.OneHot()
        self.one_hot_masks = self.transform(self.masks)
        self.n_masks = self.one_hot_masks.shape[0] - 1
        self.ct_files = natsorted([os.path.join(ct_path, ct) for ct in os.listdir(ct_path)
                                   if ct.endswith("dcm")])
        self.ds_ct = ds_cts or [dcmread(ct_file, force=True) for ct_file in self.ct_files]

        self.n_slices = self.one_hot_masks.spatial_shape[2]

        self.image_orientation_patient = list(self.get_dicom_value('ImageOrientationPatient', 0))
        self.pixel_spacing = list(self.get_dicom_value('PixelSpacing', 0))

    def __str__(self):
        message = f"Structure(s): {str(self.masks)}"
        return message

    def get_dicom_value(self, tag, n_slice=0):
        """ Return a dicom tag value.

        :param tag: Dicom tag in keyword format.
        :type tag: string
        :param n_slice: Slice number.
        :type n_slice: int
        :return: the actual value. A regular value like a number or string (or list of them), or a Sequence.
        :rtype: :class:`pydicom.dataelem.DataElement`
        """
        return self.ds_ct[n_slice][Tag(tag)].value

    def coordinate_mapping(self, c, r, image_position_patient):
        """The mapping of pixel location (c,r) to Reference Coordinate System (RCS).

        :param c: column.
        :type c: float
        :param r: row.
        :type r: float
        :param image_position_patient: The x, y, and z coordinates of the
            upper left hand corner of the image, in mm.
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

    def coordinates(self, mask):
        """Give the coordinates of the corresponding structures in the RCS and the SOP Instance UID.

        :param mask: mask.
        :type mask: :class:`torch.Tensor`
        :return: List of ROI contour sequence.
        :rtype: list[(str,list[int])]
        """
        mask = mask.numpy().astype(bool)
        referenced_contour_data = []
        self.ct_files.reverse()
        for i in range(mask.shape[-1]):
            # removing holes using a large value to be sure all the holes are removed
            img = morphology.remove_small_holes(mask[..., i], 100000, in_place=True)
            contours = find_contours(img)
            for contour in contours:
                if len(contour):
                    x = np.array(contour[:, 1])
                    y = np.array(contour[:, 0])
                    n_points = len(x)
                    # s is how much we want the spline to stick the points. If too high, interpolation 'moves away'
                    # from the real outline. If too small, it creates a crenellation
                    # ToDo check per=False
                    tck = interpolate.splprep([x, y], per=True, s=n_points // 10.)
                    xi, yi = interpolate.splev(tck[1], tck[0])

                    contour = list(zip(xi, yi))
                    image_position_patient = self.get_dicom_value('ImagePositionPatient', i)
                    mask_coordinates = []
                    for coord in contour:
                        r, c = coord
                        x, y, z = self.coordinate_mapping(c, r, image_position_patient)
                        mask_coordinates.append(round(x, 4))
                        mask_coordinates.append(round(y, 4))
                        mask_coordinates.append(round(z, 4))
                    referenced_contour_data.append((self.ds_ct[i].SOPInstanceUID, mask_coordinates))

        return referenced_contour_data
