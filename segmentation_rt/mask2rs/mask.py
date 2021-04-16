import os

import numpy as np
import torchio as tio
from natsort import natsorted
from pydicom import dcmread
from scipy import interpolate
from skimage import morphology
from skimage.measure import find_contours


class Mask:
    """
    Class that hold any structures aligned with a reference CT.

    :param tio.LABEL mask:
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
        self.masks_itk = self.masks.as_sitk()
        self.transform = tio.OneHot()
        self.one_hot_masks = self.transform(self.masks)
        self.n_masks = self.one_hot_masks.shape[0] - 1
        self.ct_files = natsorted([os.path.join(ct_path, ct) for ct in os.listdir(ct_path)
                                   if ct.endswith("dcm")])
        self.ds_ct = ds_cts or [dcmread(ct_file, force=True) for ct_file in self.ct_files]

    def __str__(self):
        message = f"Structure(s): {str(self.masks)}"
        return message

    def coordinates(self, mask):
        """
        Give the coordinates of the corresponding structures in the RCS and the SOP Instance UID.

        :param mask: mask.
        :type mask: :class:`torch.Tensor`
        :return: List of ROI contour sequence.
        :rtype: list[(str,list[int])]
        """
        mask = mask.numpy().astype(bool)
        referenced_contour_data = []
        self.ds_ct.reverse()
        for s in range(mask.shape[-1]):
            # removing holes using a large value to be sure all the holes are removed
            img = morphology.remove_small_holes(mask[..., s], 100000, in_place=True)
            contours = find_contours(img)
            for contour in contours:
                if len(contour):
                    x = np.array(contour[:, 1])
                    y = np.array(contour[:, 0])
                    n_points = len(x)
                    # s is how much we want the spline to stick the points. If too high, interpolation 'moves away'
                    # from the real outline. If too small, it creates a crenellation
                    # ToDo check per=False
                    if n_points > 3:
                        tck = interpolate.splprep([x, y], per=True, s=n_points // 10.)
                        xi, yi = interpolate.splev(tck[1], tck[0])

                        contour = list(zip(xi, yi))
                        mask_coordinates = []
                        for coord in contour:
                            r, c = coord
                            x, y, z = self.masks_itk.TransformContinuousIndexToPhysicalPoint([c, r, s])
                            mask_coordinates.append(round(x, 4))
                            mask_coordinates.append(round(y, 4))
                            mask_coordinates.append(round(z, 4))
                        referenced_contour_data.append((self.ds_ct[s].SOPInstanceUID, mask_coordinates))

        return referenced_contour_data
