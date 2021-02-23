import functools

import torch.nn as nn


def build_conv_block(dim, norm_layer, use_dropout, use_bias):
    """Construct a convolutional block.

    :param dim: the number of channels in the conv layer.
    :type dim: int
    :param norm_layer: normalization layer.
    :type norm_layer: :class:`nn.Module`
    :param use_dropout: if use dropout layers.
    :type use_dropout: bool
    :param use_bias: if the conv layer uses bias or not
    :type use_bias: bool
    :return: Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
    :rtype: :class:`nn.Sequential`
    """
    conv_block = []
    conv_block += [nn.ReplicationPad2d(1)]
    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    conv_block += [nn.ReLU(True)]

    if use_dropout:
        conv_block += [nn.Dropout(0.5)]

    conv_block += [nn.ReplicationPad2d(1)]
    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    conv_block += [norm_layer(dim)]

    return nn.Sequential(*conv_block)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downscaling/up-sampling operations.

    :param int input_nc:
        the number of channels in input images.

    :param int output_nc:
        the number of channels in output images.

    :param int ngf:
        the number of filters in the last conv layer.

    :param int n_blocks:
        the number of ResNet blocks.

    :param norm_layer:
        normalization layer.

    :param bool use_dropout:
        if use dropout layers.

    :param list[int] gpu_ids:
        gpu id.
    """

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9,
                 norm_layer=functools.partial(nn.BatchNorm2d, affine=True),
                 use_dropout=False, gpu_ids=None):

        if gpu_ids is None:
            gpu_ids = []
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2 * ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(2 * ngf),
            nn.ReLU(True),

            nn.Conv2d(2 * ngf, 4 * ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4 * ngf),
            nn.ReLU(True),
        ]

        for i in range(n_blocks):
            model += [ResnetBlock(4 * ngf, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=True)]

        model += [

            nn.ConvTranspose2d(4 * ngf, 2 * ngf,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=True),
            norm_layer(2 * ngf),
            nn.ReLU(True),

            nn.Conv2d(2 * ngf, ngf, kernel_size=3, padding=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        if len(self.gpu_ids) > 1:
            return nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        else:
            return self.model(x)

    def __str__(self):
        return "ResnetGenerator"


class ResnetBlock(nn.Module):
    """Initialize the Resnet block
    A resnet block is a conv block with skip connections
    We construct a conv block with build_conv_block function,
    and implement skip connections in <forward> function.
    Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf.

     :param int dim:
        the number of channels in the conv layer.

    :param norm_layer:
        normalization layer.

    :param bool use_dropout:
        if use dropout layers.

    :param bool use_bias:
        if use bias.
    """

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = build_conv_block(dim, norm_layer, use_dropout, use_bias)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        """Standard forward"""
        out = self.conv_block(x)
        out = self.relu(x + out)
        return out
