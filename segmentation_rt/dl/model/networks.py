import functools

import torch.nn as nn


def dice_loss(output, target):
    """
    Dice Loss between two tensors.

    :param output: input tensor.
    :type output: :class:`torch.Tensor`
    :param target: ground truth tensor.
    :type target: :class:`torch.Tensor`
    :return: dice loss.
    """
    smooth = 1.
    loss = 0.
    for c in range(target.shape[1]):
        output_flat = output[:, c].contiguous().view(-1)
        target_flat = target[:, c].contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss += 1. - ((2. * intersection + smooth) /
                      (output_flat.sum() + target_flat.sum() + smooth))

    return loss / target.shape[1]


def dice_score(output, target):
    """
    Dice Score between two tensors.

        :param output: input tensor.
        :type output: :class:`torch.Tensor`
        :param target: ground truth tensor.
        :type target: :class:`torch.Tensor`
        :return: dice score.
        """
    return 1 - dice_loss(output, target)


def weights_init(m):
    """
    Initialize network weights.

    :param m: Module
    :type m: :class:`torch.nn.Module`
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def build_conv_block(dim, norm_layer, use_dropout, use_bias):
    """
    Construct a convolutional block.

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
    conv_block += [nn.ReplicationPad3d(1)]
    conv_block += [nn.Conv3d(dim, dim, kernel_size=3, bias=use_bias)]
    conv_block += [nn.ReLU(True)]

    if use_dropout:
        conv_block += [nn.Dropout(0.5)]

    conv_block += [nn.ReplicationPad3d(1)]
    conv_block += [nn.Conv3d(dim, dim, kernel_size=3, bias=use_bias)]
    conv_block += [norm_layer(dim)]

    return nn.Sequential(*conv_block)


class ResnetGenerator(nn.Module):
    """
    Resnet-based generator that consists of Resnet blocks between a few downscaling/up-sampling operations.

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
    """

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9,
                 norm_layer=functools.partial(nn.BatchNorm3d, affine=True),
                 use_dropout=False):

        super(ResnetGenerator, self).__init__()

        model = [
            nn.ReplicationPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv3d(ngf, 2 * ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(2 * ngf),
            nn.ReLU(True),

            nn.Conv3d(2 * ngf, 4 * ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4 * ngf),
            nn.ReLU(True),
        ]

        for _ in range(n_blocks):
            model += [ResnetBlock(4 * ngf, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=True)]

        model += [

            nn.ConvTranspose3d(4 * ngf, 2 * ngf,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=True),
            norm_layer(2 * ngf),
            nn.ReLU(True),

            nn.Conv3d(2 * ngf, ngf, kernel_size=3, padding=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv3d(ngf, output_nc, kernel_size=7, padding=3),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)

    def __str__(self):
        return "ResnetGenerator"


class ResnetBlock(nn.Module):
    """
    Initialize the Resnet block.

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
        """Standard forward."""
        out = self.conv_block(x)
        out = self.relu(x + out)
        return out


def define_generator(input_nc, output_nc, ngf, n_blocks, device,
                     use_dropout=False):
    """
    Create a generator.

    :param input_nc: the number of channels in input images.
    :type input_nc: int
    :param output_nc: the number of channels in output images.
    :type output_nc: i
    :param ngf: the number of filters in the last conv layer.
    :type ngf: int
    :param n_blocks: the number of ResNet blocks.
    :type n_blocks: int
    :param use_dropout: if use dropout layers.
    :type use_dropout: bool
    :param device:  device.
    :type device:
    :return: a generator.
    :rtype: :class:`torch.nn.Module`
    """

    norm_layer = functools.partial(nn.BatchNorm3d, affine=True)

    net_g = ResnetGenerator(input_nc, output_nc, ngf, n_blocks, norm_layer=norm_layer,
                            use_dropout=use_dropout)

    net_g.to(device)
    net_g.apply(weights_init)
    return net_g


def print_network(net, out_f=None):
    """
    Prints the number of learnable parameters.

    :param net: the network.
    :type net:
    :param out_f: file
    :type out_f:
    :return: number of learnable parameters.
    :rtype: int
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()
    return num_params
