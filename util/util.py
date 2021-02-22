import os

import numpy as np
import png


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


def save_image(image, path, start=None, end=None):
    if start and end:
        slicer = range(start, end)
    else:
        slicer = range(image.shape[2])
    ipp = os.path.basename(os.path.dirname(path))
    for i in slicer:
        filename = os.path.join(path, ipp + "_" + str(i))
        with open(filename + ".png", 'wb') as f:
            writer = png.Writer(width=image.shape[0], height=image.shape[1], bitdepth=16, greyscale=True)
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
