import os

import numpy as np
import png


def print_log(out_f, message):
    """Writes in log file.

    :param out_f:
    :type out_f:
    :param message:
    :type message:
    """
    out_f.write(message + "\n")
    out_f.flush()
    print(message)


def format_log(epoch, i, errors, t, prefix=True):
    message = '(epoch: %d, iteration: %d, time: %.3f) ' % (epoch, i, t)
    if not prefix:
        message = ' ' * len(message)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message


def parse_gpu_ids(gpu_ids):
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            gpu_ids.append(gpu_id)
    return gpu_ids


def listdir_full_path(path):
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


def parse_opt_file(opt_path):
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

    opt = None
    with open(opt_path) as f:
        opt = dict()
        for line in f:
            if line.startswith('-----'):
                continue
            k, v = line.split(':')
            opt[k.strip()] = parse_val(v.strip())
    return opt

