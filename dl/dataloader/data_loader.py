import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class AlignedDataset(data.Dataset):
    """A dataset class for paired image dataset. Class that overrides :class:`data.Dataset`.

    It assumes that the directory '/root/train' contains image pairs in the form of {ct,mask_name}.
    During test time, you need to prepare a directory '/root/test'.

    :param string root:
        Root directory.

    :param string mask_name:
        Name of the mask.

    :param int load_size:
        Load images at this size.

    :param int crop_size:
        Crop images at this size.

    :param bool flip:
        Flip images or not.

    :param int crop_size:
        Crop images at this size.

    :param str subset:
        train or test.
    """

    def __init__(self, root, mask_name, load_size, crop_size, flip, subset):

        self.root = root
        self.mask = mask_name
        self.subset = subset
        self.crop_size = crop_size
        self.load_size = load_size
        self.flip = flip

        if subset == 'training':
            self.dir_image = os.path.join(self.root, 'train')

        elif subset == 'testing':
            self.dir_image = os.path.join(self.root, 'test')

        else:
            raise NotImplementedError('subset %s no supported' % subset)

        self.image_names = sorted(os.listdir(os.path.join(self.dir_image, "ct")))
        self.ct_dir = os.path.join(self.dir_image, "ct")
        self.mask_dir = os.path.join(self.dir_image, self.mask)

        self.ct_paths = sorted(make_dataset(self.ct_dir))
        self.mask_paths = sorted(make_dataset(self.mask_dir))

        # shuffle data
        rand_state = random.getstate()
        random.seed(123)
        index = range(len(self.ct_paths))
        random.shuffle(list(index))
        self.ct_paths = [self.ct_paths[i] for i in index]
        self.mask_paths = [self.mask_paths[i] for i in index]

        random.setstate(rand_state)

        self.size = len(self.ct_paths)

    def __getitem__(self, index):
        ct_path = self.ct_paths[index]
        mask_path = self.mask_paths[index]

        ct_img = Image.open(ct_path).convert('I')
        mask_img = Image.open(mask_path).convert('I')

        parameters = get_transform_parameters(self.load_size, self.crop_size, self.flip)
        transform = get_transform(parameters)

        ct_img = transform(ct_img)
        ct_torch = torch.from_numpy(np.array(ct_img, dtype=np.float32))
        ct_torch = ct_torch.clamp(0, 2500) / 1250. - 1.

        mask_img = transform(mask_img)
        mask_torch = torch.from_numpy(np.array(mask_img, dtype=np.float32))

        dataset = OrderedDict([('patientID', ct_path),
                               ('ct', ct_torch.type(torch.float32).unsqueeze_(0)),
                               ('mask', mask_torch.unsqueeze_(0))
                               ])

        return dataset

    def __len__(self):
        return self.size


class SingleDataset(data.Dataset):
    """This dataset class can load a set of images specified by the path.

    It assumes that the directory '/root/' contains CT images.

    :param string root:
        Root directory.

    :param string mask_name:
        Name of the mask.

    :param int load_size:
        Load images at this size.
    """

    def __init__(self, root, mask_name, load_size):

        self.root = root
        self.mask = mask_name
        self.load_size = load_size

        self.ct_paths = sorted(make_dataset(self.root))
        self.size = len(self.ct_paths)

    def __getitem__(self, index):
        ct_path = self.ct_paths[index]

        ct_img = Image.open(ct_path).convert('I')

        parameters = get_transform_parameters(self.load_size, self.load_size, False)
        transform = get_transform(parameters)

        ct_img = transform(ct_img)
        ct_torch = torch.from_numpy(np.array(ct_img, dtype=np.float32))
        ct_torch = ct_torch.clamp(0, 2500) / 1250. - 1.

        dataset = OrderedDict([('ct_path', ct_path),
                               ('ct', ct_torch.type(torch.float32).unsqueeze_(0))
                               ])

        return dataset

    def __len__(self):
        return self.size


class DataLoader(object):
    """Dataset.

        :param string root:
            Root directory.

        :param string mask_name:
            Name of the mask.

        :param str subset:
            train or test.

        :param int batch_size:
            how many samples per batch to load (default: 1).

        :param int load_size:
            Load images at this size(default: 512).

        :param int crop_size:
            Crop images at this size(default: 256).

        :param bool flip:
            Flip images or not (default: False).

        :param bool shuffle:
            set to True to have the data reshuffled at every epoch (default: False).

        :param bool drop_last:
            set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
            (default: False)

        """

    def __init__(self, root, mask_name, subset, batch_size=1, load_size=512, crop_size=512, flip=False,
                 shuffle=False, drop_last=False):
        self.subset = subset
        if self.subset == "prediction":
            self.dataset = SingleDataset(root, mask_name, load_size)
        else:
            self.dataset = AlignedDataset(root, mask_name, load_size, crop_size, flip, subset)

        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=1,
            drop_last=drop_last)

    def __len__(self):
        return len(self.dataset)

    def load_data(self):
        print(f"#{self.subset} images = {self.__len__()}")
        return self.dataloader


def get_transform_parameters(crop_size, load_size, flip):
    """Parameters for data augmentation.

    :param crop_size: Crop size.
    :type crop_size: int
    :param load_size: Image rescaling.
    :type load_size: int
    :param flip: Flip or not
    :type flip: bool
    :return: Crop position and flip.
    :rtype: dict[int, int, (int, int), bool]
    """
    new_h = new_w = load_size

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    if flip:
        flip = random.random() > 0.5
    else:
        flip = False

    return {'load_size': load_size, 'crop_size': crop_size, 'crop_position': (x, y), 'flip': flip}


# noinspection PyTypeChecker
def get_transform(parameters):
    transform_list = [transforms.Resize([parameters['load_size'], parameters['load_size']], Image.NEAREST)]

    if parameters['load_size'] != parameters['crop_size']:
        transform_list.append(transforms.Lambda(lambda img:
                                                __crop(img, parameters['crop_position'], parameters['crop_size'])))
        transform_list.append(transforms.Lambda(lambda img:
                                                __flip(img, parameters['flip'])))

    return transforms.Compose(transform_list)


def make_dataset(directory):
    """List all images in a directory.

    :param directory: path to directory.
    :type directory: str
    :return: List of file in directory.
    :rtype: list[str]
    """
    images = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory

    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)

    return images


def __crop(image, position, size):
    """Return a cropped image.

    :param image: an image.
    :type image: :class:`Image`
    :param position: origin (x, y).
    :type position: (int, int)
    :param size: desired size.
    :type size: int
    :return: Image cropped.
    :rtype: :class:`Image`
    """
    ow, oh = image.size
    x1, y1 = position
    tw = th = size
    if ow > tw or oh > th:
        return image.crop((x1, y1, x1 + tw, y1 + th))
    return image


def __flip(image, flip):
    """Return a flipped or not image.

    :param image: an image.
    :type image: :class:`Image`
    :param flip: flip or not.
    :type flip: bool
    :return: image flipped or not.
    :rtype: :class:`Image`
    """
    if flip:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image
