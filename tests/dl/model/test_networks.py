import os.path
import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from segmentation_rt.dl.model.networks import *

torch.manual_seed(0)


class TestNetworks(TestCase):

    def setUp(self):
        self.input = torch.randn((1, 1, 9, 25, 25))
        self.output = torch.ones((1, 2, 2, 2, 2))
        self.target = torch.zeros((1, 2, 2, 2, 2))

    def test_dice_loss(self):
        loss = float(dice_loss(self.output, self.target).sum())
        self.assertEqual(loss, 0.8888888955116272)

    def test_dice_score(self):
        score = float(dice_score(self.output, self.target))
        self.assertEqual(score, 0.1111111044883728)

    def test_build_conv_block(self):
        conv_block = build_conv_block(1, torch.nn.BatchNorm3d, False, True)
        conv_block.apply(weights_init)
        self.assertEqual(conv_block(self.input).sum(), conv_block(self.input).sum())
        conv_block = build_conv_block(1, torch.nn.BatchNorm3d, True, True)
        self.assertNotEqual(conv_block(self.input).sum(), conv_block(self.input).sum())

    def test_resnet(self):
        resnet = ResnetGenerator(1, 3, ngf=4, n_blocks=3)
        resnet.apply(weights_init)
        self.assertEqual(resnet.forward(self.input).sum(), resnet.forward(self.input).sum())
        resnet = ResnetGenerator(1, 3, ngf=4, n_blocks=3, use_dropout=True)
        self.assertNotEqual(resnet.forward(self.input).sum(), resnet.forward(self.input).sum())

    def test_resblock(self):
        resblock = ResnetBlock(1, torch.nn.BatchNorm3d, False, True)
        resblock.apply(weights_init)
        self.assertEqual(resblock.forward(self.input).sum(), resblock.forward(self.input).sum())
        resblock = ResnetBlock(1, torch.nn.BatchNorm3d, True, True)
        self.assertNotEqual(resblock.forward(self.input).sum(), resblock.forward(self.input).sum())

    def test_print_network(self):
        conv_block = build_conv_block(1, torch.nn.BatchNorm3d, False, True)
        with open(os.path.join(Path(tempfile.gettempdir()), "net.txt"), 'w') as f:
            num_params = print_network(conv_block, f)
        self.assertEqual(num_params, 58)
