import functools

import torch.nn as nn
from segmentation_rt.dl.model.modules import ResnetGenerator


def dice_loss(output, target):
    """Dice Loss between two tensors.

    :param output: input tensor.
    :type output: :class:`torch.Tensor`
    :param target: ground truth tensor.
    :type target: :class:`torch.Tensor`
    :return: dice loss.
    """
    smooth = 1.
    loss = 0.
    for c in range(target.shape[1]):
        oflat = output[:, c].contiguous().view(-1)
        tflat = target[:, c].contiguous().view(-1)
        intersection = (oflat * tflat).sum()
        loss += 1. - ((2. * intersection + smooth) /
                      (oflat.sum() + tflat.sum() + smooth))

    return loss / target.shape[1]


def dice_score(output, target):
    """Dice Score between two tensors.

        :param output: input tensor.
        :type output: :class:`torch.Tensor`
        :param target: ground truth tensor.
        :type target: :class:`torch.Tensor`
        :return: dice score.
        """
    return 1 - dice_loss(output, target)


def weights_init(m):
    """Initialize network weights.

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


def define_generator(input_nc, output_nc, ngf, n_blocks, device,
                     use_dropout=False):
    """Create a generator.

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
    """Prints the number of learnable parameters.

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
