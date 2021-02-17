import os
from collections import OrderedDict

import torch
from torch.nn.functional import binary_cross_entropy

import dl.model.networks as networks


class Model(object):
    """Initialize chosen model """

    def __init__(self, expr_dir, testing=False):

        self.expr_dir = expr_dir
        self.input_nc = 1
        self.output_nc = 1
        self.ngf = 64
        self.n_blocks = 9
        self.use_dropout = False
        self.gpu_ids = '0'
        self.lr = 0.0002
        self.old_lr = self.lr
        self.niter_decay = 100
        self.niter = 100

        self.norm = "batch"
        self.beta1 = 0.5
        self.monitor_gnorm = True
        self.max_gnorm = 500.

        # define all networks we need here
        self.netG = networks.define_generator(input_nc=self.input_nc,
                                              output_nc=self.output_nc, ngf=self.ngf,
                                              n_blocks=self.n_blocks,
                                              use_dropout=self.use_dropout,
                                              gpu_ids=self.gpu_ids)

        # define all optimizers here
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=self.lr, betas=(self.beta1, 0.999))

        self.criterion = binary_cross_entropy

        if not testing:
            num_params = 0
            with open("%s/nets.txt" % self.expr_dir, 'w') as nets_f:
                num_params += networks.print_network(self.netG, nets_f)
                nets_f.write('# parameters: %d\n' % num_params)
                nets_f.flush()

    def train_instance(self, ct, segmentation):

        fake_segmentation = self.netG.forward(ct)
        loss = 1000 * self.criterion(fake_segmentation, segmentation)

        self.optimizer_G.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.max_gnorm)
        self.optimizer_G.step()

        losses = OrderedDict([('L_global', loss.data.item())])

        segmentation = torch.unsqueeze(segmentation, 1) * 2500
        fake_segmentation = torch.unsqueeze(fake_segmentation, 1) * 2500
        visuals = OrderedDict([('ct', ct.data),
                               ('segmentation_mask', segmentation.data),
                               ('fake_segmentation_mask', fake_segmentation.data)
                               ])
        if self.monitor_gnorm:
            gnorms = OrderedDict([('gnorm', gnorm)])

            return losses, visuals, gnorms

        return losses, visuals

    def synthesize(self, ct):
        fake_segmentation = self.netG.forward(ct)
        fake_segmentation = torch.unsqueeze(fake_segmentation, 1) * 2500
        visuals = OrderedDict([('fake_segmentation_mask', fake_segmentation.data)])
        return visuals

    def update_learning_rate(self):
        lrd = self.lr / self.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, chk_name):
        chk_path = os.path.join(self.expr_dir, chk_name)
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
