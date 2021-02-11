import os

import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from pydicom import dcmread
from pydicom.tag import Tag
from skimage.measure import find_contours, approximate_polygon


class Mask:

    def __init__(self, path, ds_ct=None):
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
        self.ds_ct = ds_ct or [dcmread(ct_file, force=True) for ct_file in self.ct_files]

        self.n_slices = len(self.masks_files)

        self.image_orientation_patient = self.get_dicom_value('ImageOrientationPatient', 0)
        self.pixel_spacing = self.get_dicom_value('PixelSpacing', 0)

    def __str__(self):
        message = f"Structure(s): {', '.join(self.masks)}"
        return message

    def get_dicom_value(self, tag, n_slice=0):
        return self.ds_ct[n_slice][Tag(tag)].value

    def coord_calculation(self, c, r, image_position_patient):
        sx, sy, sz = np.array(image_position_patient, dtype=np.float32)
        delta_r, delta_c = np.array(self.pixel_spacing[:], dtype=np.float32)

        xx, xy, xz = np.array(self.image_orientation_patient[:3], dtype=np.float32)
        yx, yy, yz = np.array(self.image_orientation_patient[3:], dtype=np.float32)

        t_1 = np.array(self.get_dicom_value('ImagePositionPatient', 0)[:])
        t_n = np.array(self.get_dicom_value('ImagePositionPatient', self.n_slices - 1)[:])

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
        referenced_contour_data = []
        self.ct_files.reverse()
        for index, png in enumerate(self.masks_files):
            img = plt.imread(os.path.join(self.masks_path, mask_name, png))
            # ToDo validate 0
            contour = find_contours(img, 0)
            if len(contour):
                coords = approximate_polygon(contour[0], tolerance=0.05)
                image_position_patient = self.get_dicom_value('ImagePositionPatient', index)
                mask_coordinates = []
                for coord in coords:
                    r, c = coord
                    x, y, z = self.coord_calculation(c, r, image_position_patient)
                    mask_coordinates.append(round(x, 4))
                    mask_coordinates.append(round(y, 4))
                    mask_coordinates.append(round(z, 4))
                referenced_contour_data.append((self.ds_ct[index].SOPInstanceUID, mask_coordinates))

        return referenced_contour_data
