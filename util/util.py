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
