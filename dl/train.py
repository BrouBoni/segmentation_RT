#!/usr/bin/env python

import os
import random
import time

import numpy as np
import png
import torch
import torchvision.utils as vutils
from torch.autograd import Variable

from dataloader.data_loader import DataLoader
from model.model import JulienGNet
from options.dl_option import TrainOptions


def print_log(out_f, message):
    out_f.write(message + "\n")
    out_f.flush()
    print(message)


def format_log(epoch, i, errors, t, prefix=True):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    if not prefix:
        message = ' ' * len(message)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message


def train_model():
    opt = TrainOptions().parse(sub_dirs=["train_vis_cycle"])
    out_f = open("%s/results.txt" % opt.expr_dir, 'w')
    use_gpu = len(opt.gpu_ids) > 0

    if opt.seed is not None:
        print("using random seed:", opt.seed)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        if use_gpu:
            torch.cuda.manual_seed_all(opt.seed)

    train_data_loader = DataLoader(opt, subset='train', batch_size=opt.batchSize)
    # test_data_loader = DataLoader(opt, subset='test', batch_size=1, drop_last=True)

    train_dataset = train_data_loader.load_data()
    dataset_size = len(train_data_loader)
    print_log(out_f, '#training images = %d' % dataset_size)

    # test_dataset = test_data_loader.load_data()
    # print_log(out_f, '#test images = %d' % len(test_data_loader))

    model = JulienGNet(opt)

    total_steps = 0
    print_start_time = time.time()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataset):
            ct = Variable(data['ct'])
            mask = Variable(data['mask'])

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            if use_gpu:
                ct = ct.cuda()
                mask = mask.cuda()

            if opt.monitor_gnorm:
                losses, visuals, gnorms = model.train_instance(ct, mask)
            else:
                losses, visuals = model.train_instance(ct, mask)

            if total_steps % opt.print_freq == 0:
                t = (time.time() - print_start_time) / opt.batchSize
                print_log(out_f, format_log(epoch, epoch_iter, losses, t))
                if opt.monitor_gnorm:
                    print_log(out_f, format_log(epoch, epoch_iter, gnorms, t, prefix=False) + "\n")

                print_start_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print_log(out_f, 'saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
            model.save('latest')

        print_log(out_f, 'End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()

    return 1


if __name__ == "__main__":
    train_model()
