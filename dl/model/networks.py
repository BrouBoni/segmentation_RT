import functools

import torch
import torch.nn as nn

from dl.model.modules import ResnetGenerator


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
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_generator(input_nc, output_nc, ngf, n_blocks,
                     use_dropout=False, gpu_ids=None):
    """Create a generator.

    :param input_nc: the number of channels in input images.
    :type input_nc: int
    :param output_nc: the number of channels in output images.
    :type output_nc: int
    :param ngf: the number of filters in the last conv layer.
    :type ngf: int
    :param n_blocks: the number of ResNet blocks.
    :type n_blocks: int
    :param use_dropout: if use dropout layers.
    :type use_dropout: bool
    :param gpu_ids:  gpu id.
    :type gpu_ids: str
    :return: a generator.
    :rtype: :class:`torch.nn.Module`
    """

    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())

    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

    net_g = ResnetGenerator(input_nc, output_nc, ngf, n_blocks, norm_layer=norm_layer,
                            use_dropout=use_dropout, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        net_g.cuda()
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
    # print(net)
    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()
    return num_params
