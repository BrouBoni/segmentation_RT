import functools

import torch
import torch.nn as nn
from model.modules import ResnetBlock, InstanceNorm2d


###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(InstanceNorm2d, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, n_blocks, norm='batch',
             use_dropout=False, gpu_ids=[]):
    netG = None
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())

    norm_layer = get_norm_layer(norm_type=norm)

    netG = ResnetGenerator(input_nc, output_nc, ngf, n_blocks, norm_layer=norm_layer,
                           use_dropout=use_dropout, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG

def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()
    return num_params

######################################################################
# ResnetGenerator for deterministic mappings
######################################################################
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=InstanceNorm2d, use_dropout=False,
                 gpu_ids=[], padding_type='reflect'):
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
            model += [ResnetBlock(4 * ngf, padding_type=padding_type, norm_layer=norm_layer,
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

    def forward(self, input):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
