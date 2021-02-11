import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class Ct2Struct(object):

    def __init__(self, opt, subset):

        self.root = opt.dataroot
        self.mask = opt.mask
        self.subset = subset
        self.opt = opt
        self.cropSize = opt.cropSize
        self.loadSize = (opt.loadSize, opt.loadSize)

        if subset == 'train':
            self.dir_image = os.path.join(self.root, 'train')

        elif subset == 'test':
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
        #ToDo change
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

        params = get_params(self.cropSize, self.loadSize)
        transform = get_transform(self.opt, params)

        ct_img = transform(ct_img)
        ct_img = torch.from_numpy(np.array(ct_img, np.int32, copy=False)).type(torch.float32)
        ct_img = ct_img.clamp(0, 2500) / 1250. - 1.

        mask_img = transform(mask_img).type(torch.float32)

        data = OrderedDict([('patientID', self.ct_paths[index]),
                            ('ct',  torch.as_tensor(ct_img)),
                            ('mask', mask_img)
                            ])

        return data

    def __len__(self):
        return self.size


class DataLoader(object):
    def __init__(self, opt, subset, batch_size,
                 shuffle=False, drop_last=False):
        self.opt = opt
        self.dataset = Ct2Struct(opt, subset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(1),
            drop_last=drop_last)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)


def get_params(cropSize, loadSize):
    new_h, new_w = loadSize

    x = random.randint(0, np.maximum(0, new_w - cropSize))
    y = random.randint(0, np.maximum(0, new_h - cropSize))

    flip = random.random() > 0.5
    flip = False
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params):
    transform_list = [transforms.Resize([opt.loadSize, opt.loadSize], Image.NEAREST)]

    if opt.crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.cropSize)))
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)

    return images