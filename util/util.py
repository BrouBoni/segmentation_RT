import os

import numpy as np
import png
import torch
import torchio as tio


def print_log(out_f, message):
    """Writes in log file.

    :param out_f: I/O stream.
    :type out_f:
    :param message: message to display.
    :type message: str
    """
    out_f.write(message + "\n")
    out_f.flush()
    print(message)


def format_log(epoch, iteration, errors, t, prefix=True):
    """Generic format for print/log.

    :param epoch: epoch.
    :type epoch: int
    :param iteration: iteration.
    :type iteration: int
    :param errors: errors dictionary.
    :type errors: dict[float]
    :param t: time.
    :type t: float
    :param prefix: if prefix.
    :type prefix: str
    :return: message.
    :rtype: str.
    """
    message = '(epoch: %d, iteration: %d, time: %.3f) ' % (epoch, iteration, t)
    if not prefix:
        message = ' ' * len(message)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message


def parse_gpu_ids(gpu_ids):
    """Parse gpu device.

    :param gpu_ids: "-1" for cpu, "0" for the first gpu. "1,3" for the second and fourth gpu.
    :type gpu_ids: str
    :return: list of devices.
    :rtype: list[int]
    """
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            gpu_ids.append(gpu_id)
    return gpu_ids


def listdir_full_path(path):
    """List files in path and return their absolute path.

    :param path: path.
    :type path: str
    :return: list of absolute path.
    :rtype: list[str]
    """
    return [os.path.join(path, f) for f in os.listdir(path)]


def save_image(image, path, width=10, bit_depth=8, start=None, end=None):
    """Save image from numpy array.

    :param bit_depth: encoding.
    :type bit_depth: int
    :param image: 3D npy array
    :type image:
    :param path: output folder.
    :type path: str
    :param start: first slice.
    :type path: int
    :param end: last slice.
    :type path: int
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
            writer = png.Writer(width=image.shape[0], height=image.shape[1], bitdepth=bit_depth, greyscale=True)
            array = image[:, :, i].astype(np.uint16)
            array2list = array[:, :].tolist()
            writer.write(f, array2list)


def save_png(array, path):
    """Save array as png.

    :param array: Numpy array.
    :type array:
    :param path: name of the file.
    :type path: str
    """
    with open(path, 'wb') as f:
        writer = png.Writer(width=array.shape[1], height=array.shape[0], bitdepth=16, greyscale=True)
        array = array.astype(np.uint16)
        array2list = array[:, :].tolist()
        writer.write(f, array2list)


def get_subjects(path, structures, transform):
    """Browse the path folder to build a dataset. Folder must contains Subjects with the CT and masks.

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


def parse_opt_file(opt_path):
    """Parse option at opt_path, Return option.

    :param opt_path: option path.
    :type opt_path: str
    """

    def parse_val(s):
        if s == 'None':
            return None
        if s == 'True':
            return True
        if s == 'False':
            return False
        if s == 'inf':
            return float('inf')
        try:
            float_value = float(s)
            # special case
            if '.' in s:
                return float_value
            i = int(float_value)
            return i if i == float_value else float_value
        except ValueError:
            return s

    with open(opt_path) as f:
        opt = dict()
        for line in f:
            if line.startswith('-----'):
                continue
            k, v = line.split(':')
            opt[k.strip()] = parse_val(v.strip())
    return opt
