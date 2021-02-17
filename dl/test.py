
import os
import random

import numpy as np
import png
import torch
from torch.autograd import Variable

from dataloader.data_loader import DataLoader
from model.model import JulienGNet
from options.dl_option import TestOptions, parse_opt_file


def print_log(out_f, message):
    out_f.write(message + "\n")
    out_f.flush()
    print(message)

def visualize_pred(opt, ct, segmentation, index):
    size = ct.size()

    images = [img.cpu().unsqueeze(1) for img in segmentation.values()]
    labels = [label for label in segmentation.keys()]
    for i, (label, image) in enumerate(zip(labels, images)):
        save_path = os.path.join(opt.res_dir, label, str(index)+'.png')
        im = image[0][0][0].numpy().astype(np.uint16)
        with open(save_path, 'wb') as f:
            writer = png.Writer(width=size[2], height=size[3], bitdepth=16, greyscale=True)
            array = im
            array2list = array[:, :].tolist()
            writer.write(f, array2list)

def test_model():
    opt = TestOptions().parse()
    dataroot = opt.dataroot

    # extract expr_dir from chk_path
    expr_dir = os.path.dirname(opt.chk_path)
    opt_path = os.path.join(expr_dir, 'opt.txt')

    # parse saved options...
    opt.__dict__.update(parse_opt_file(opt_path))
    opt.expr_dir = expr_dir
    opt.dataroot = dataroot

    # hack this for now
    opt.gpu_ids = [0]

    opt.seed = 12345
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # create results directory (under expr_dir)
    res_path = os.path.join(opt.expr_dir, opt.res_dir)
    opt.res_dir = res_path
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    use_gpu = len(opt.gpu_ids) > 0

    test_data_loader = DataLoader(opt, subset='test', batch_size=1, drop_last=True)
    test_dataset = test_data_loader.load_data()
    print('#test images = %d' % len(test_data_loader))


    model = JulienGNet(opt, testing=True)

    model.load(opt.chk_path)
    model.eval()

    for i, data in enumerate(test_dataset):
        ct = Variable(data['ct'])

        if use_gpu:
            ct = ct.cuda()

        visuals = model.synthesize(ct)
        labels = [label for label in visuals.keys()]
        # for label in labels:
        #     os.mkdir(os.path.join(opt.res_dir, label))
        visualize_pred(opt, ct, visuals, i)

if __name__ == "__main__":
    test_model()
