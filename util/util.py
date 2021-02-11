import os

import numpy as np
import png


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
