""" Implementation of useful function."""

import os

import numpy as np
import png
import torch
import torchio as tio


def print_log(out_f, message):
    """
    Writes in log file.

    :param out_f: I/O stream.
    :type out_f:
    :param message: message to display.
    :type message: str
    """
    out_f.write(message + "\n")
    out_f.flush()
    print(message)


def format_log(epoch, iteration, errors, t, prefix=True):
    """
    Generic format for print/log.

    :param epoch: epoch.
    :type epoch: int
    :param iteration: iteration.
    :type iteration: int
    :param errors: errors dictionary.
    :type errors: dict[float]
    :param t: time.
    :type t: float
    :param prefix: if prefix.
    :type prefix: bool
    :return: message.
    :rtype: str
    """
    message = '(epoch: %d, iteration: %d, time: %.3f) ' % (epoch, iteration, t)
    if not prefix:
        message = ' ' * len(message)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message


def listdir_full_path(path):
    """
    List files in path and return their absolute path.

    :param path: path.
    :type path: str
    :return: list of absolute path.
    :rtype: list[str]
    """
    return [os.path.join(path, f) for f in os.listdir(path)]


def save_image(image, path, width=10, bitdepth=8, start=None, end=None, added_slices_step=5):
    """
    Save image in png format.

    :param image: 3D npy array.
    :type image: :class:`np.ndarray`
    :param bitdepth: encoding.
    :type bitdepth: int
    :param path: output folder.
    :type path: str
    :param width: number of kept slices before the first and after the last non empty slices.
    :type width: int
    :param added_slices_step: the script saves slices from the first slice to the first non empty slices with this step
        (same at the end). if this added_slices_step ==0 , no slices are saved.
    :type added_slices_step: int
    :param start: first slice (default 0).
    :type start: int
    :param end: last slice (default -1).
    :type end: int
    """
    if start and end:
        slicer = range(start, end)
    else:
        n_slices = image.shape[2]
        # array that contains indexes of non empty slices
        non_zeros = [s for s in range(n_slices) if np.count_nonzero(image[:, :, s])]

        start = non_zeros[0] - width
        if start < 0:
            start = 0
        end = non_zeros[-1] + width
        if end > n_slices:
            end = n_slices
        slicer = np.arange(start, end, 1)

        if added_slices_step:
            step = int(added_slices_step)
            a0 = np.arange(0, start, step)
            a1 = np.arange(end, n_slices, step)
            slicer = np.concatenate((a0, slicer, a1))

    ipp = os.path.basename(os.path.dirname(path))
    for i in slicer:
        filename = os.path.join(path, ipp + "_" + str(i))
        with open(filename + ".png", 'wb') as f:
            writer = png.Writer(width=image.shape[0], height=image.shape[1], bitdepth=bitdepth, greyscale=True)
            array = image[:, :, i].astype(np.uint16)
            array2list = array[:, :].tolist()
            writer.write(f, array2list)


def get_subjects(path, structures, transform):
    """
    Browse the path folder to build a dataset. Folder must contains the subjects with the CT and masks.

    :param path: root folder.
    :type path: str
    :param structures: list of structures.
    :type structures: list[str]
    :param transform: transforms to be applied.
    :type transform: :class:`tio.transforms.Transform`
    :return: Base TorchIO dataset.
    :rtype: :class:`tio.SubjectsDataset`
    """
    subject_ids = os.listdir(path)
    subjects = []
    for subject_id in subject_ids:
        ct_path = os.path.join(path, subject_id, 'ct.nii')
        structures_path_dict = {k: os.path.join(path, subject_id, k + '.nii') for k in structures}

        subject = tio.Subject(
            ct=tio.ScalarImage(ct_path),
        )
        label_map = torch.zeros(subject["ct"].shape, dtype=torch.long)
        for i, (k, v) in enumerate(structures_path_dict.items()):
            label_map += tio.LabelMap(v).data * (i + 1)

        label_map[label_map > len(structures)] = 0
        subject.add_image(tio.LabelMap(tensor=label_map, affine=subject["ct"].affine), 'label_map')
        subjects.append(subject)

    return tio.SubjectsDataset(subjects, transform=transform)
