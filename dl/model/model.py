import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import dl.model.networks as networks


def gauss_reparametrize(mu, logvar, n_sample=1):
    """Gaussian reparametrization"""
    std = logvar.mul(0.5).exp_()
    size = std.size()
    eps = Variable(std.data.new(size[0], n_sample, size[1]).normal_())
    z = eps.mul(std[:, None, :]).add_(mu[:, None, :])
    z = torch.clamp(z, -4., 4.)
    return z.view(z.size(0) * z.size(1), z.size(2), 1, 1)


def log_prob_laplace(z, mu, log_var):
    sd = torch.exp(0.5 * log_var)
    res = - 0.5 * log_var - (torch.abs(z - mu) / sd)
    res.add_(-np.log(2))
    return res


def log_prob_gaussian(z, mu, log_var):
    res = - 0.5 * log_var - ((z - mu) ** 2.0 / (2.0 * torch.exp(log_var)))
    res = res - 0.5 * math.log(2 * math.pi)
    return res


def log_prob_gaussian_detail(z, mu, log_var, size):
    res1 = - 0.5 * log_var
    res1 = res1.view(*size).sum(2).mean(1).mean(0).data[0]
    res2 = - ((z - mu) ** 2.0 / (2.0 * torch.exp(log_var)))
    res2 = res2.view(*size).sum(2).mean(1).mean(0).data[0]
    res3 = -0.5 * math.log(2 * math.pi)

    return (res1, res2, res3 * 1 * 64 * 64)


def kld_std_guss(mu, log_var):
    """
    from Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    kld = -0.5 * torch.sum(log_var + 1. - mu ** 2 - torch.exp(log_var), dim=1)
    return kld


def discriminate(net, crit, fake, real):
    pred_fake = net(fake)
    loss_fake = 0
    for k_pred_fake in pred_fake:
        loss_fake += crit(k_pred_fake[0], False)

    pred_true = net(real)
    loss_true = 0
    for k_pred_true in pred_true:
        loss_true += crit(k_pred_true[0], True)

    return loss_fake, loss_true, pred_fake[0][0], pred_true[0][0]


def discriminate_z(net, crit, fake, real):
    pred_fake = net(fake)
    loss_fake = crit(pred_fake, False)

    pred_true = net(real)
    loss_true = crit(pred_true, True)

    return loss_fake, loss_true, pred_fake, pred_true


def criterion_GAN(pred, target_is_real, use_sigmoid=True):
    if use_sigmoid:
        if target_is_real:
            target_var = Variable(pred.data.new(pred.size()).fill_(1.))
        else:
            target_var = Variable(pred.data.new(pred.size()).fill_(0.))

        loss = F.binary_cross_entropy(pred, target_var)
    else:
        if target_is_real:
            target_var = Variable(pred.data.new(pred.size()).fill_(1.))
        else:
            target_var = Variable(pred.data.new(pred.size()).fill_(0.))

        loss = F.mse_loss(pred, target_var)

    return loss


class JulienGNet(object):
    """Cycle gan"""

    def __init__(self, opt, testing=False):

        ##### model options
        self.old_lr = opt.lr

        self.opt = opt

        ##### define all networks we need here
        self.netG = networks.define_G(input_nc=opt.input_nc,
                                          output_nc=opt.output_nc, ngf=opt.ngf,
                                          n_blocks=opt.n_blocks,
                                          norm=opt.norm, use_dropout=opt.use_dropout,
                                          gpu_ids=opt.gpu_ids)

        ##### define all optimizers here
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))

        self.criterion = F.binary_cross_entropy

        if not testing:
            num_params = 0
            with open("%s/nets.txt" % opt.expr_dir, 'w') as nets_f:
                num_params += networks.print_network(self.netG, nets_f)
                nets_f.write('# parameters: %d\n' % num_params)
                nets_f.flush()

    def train_instance(self, ct, segmentation):

        fake_segmentation = self.netG.forward(ct)
        loss = 100 * self.criterion(fake_segmentation, segmentation)

        self.optimizer_G.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.max_gnorm)
        self.optimizer_G.step()

        losses = OrderedDict([('L_global', loss.data.item())])

        segmentation = torch.unsqueeze(segmentation, 1)*2500
        fake_segmentation = torch.unsqueeze(fake_segmentation, 1)*2500
        visuals = OrderedDict([('ct', ct.data),
                               ('segmentation_mask', segmentation.data),
                               ('fake_segmentation_mask', fake_segmentation.data)
                               ])
        if self.opt.monitor_gnorm:
            gnorms = OrderedDict([('gnorm', gnorm)])

            return losses, visuals, gnorms

        return losses, visuals

    def synthetize(self, ct):
        fake_segmentation = self.netG.forward(ct)
        fake_segmentation = torch.unsqueeze(fake_segmentation, 1) * 2500
        visuals = OrderedDict([('fake_segmentation_mask', fake_segmentation.data)])
        return visuals

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, chk_name):
        chk_path = os.path.join(self.opt.expr_dir, chk_name)
        checkpoint = {
            'netG': self.netG.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict()
        }
        torch.save(checkpoint, chk_path)

    def load(self, chk_path):
        checkpoint = torch.load(chk_path)
        self.netG.load_state_dict(checkpoint['netG'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])

    def eval(self):
        self.netG.eval()

    def train(self):
        self.netG.train()
