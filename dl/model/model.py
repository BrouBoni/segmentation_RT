import os
import random
import time
from collections import OrderedDict

import numpy as np
import png
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.nn.functional import binary_cross_entropy
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import l1_loss

import dl.model.networks as networks
from util.util import print_log, format_log, parse_gpu_ids


class Model(object):
    """Initialize chosen model with parameters.
    This class allows the training of the networks and afterward for testing.

    Only resnet available, more coming soon.

    :param expr_dir: output folder.
    :type expr_dir: str
    :param seed: manual seed.
    :type seed: int
    :param gpu_ids: gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU.
    :type gpu_ids: str
    :param batch_size: input batch size.
    :type batch_size: int
    :param epoch_count: the starting epoch count.
    :type epoch_count: int
    :param niter: # of iter at starting learning rate.
    :type niter: int
    :param niter_decay: # of iter to linearly decay learning rate to zero.
    :type niter_decay: int
    :param beta1: momentum term of adam.
    :type beta1: float
    :param lr: initial learning rate for adam.
    :type lr: float
    :param ngf: # of gen filters in first conv layer.
    :type ngf: int
    :param n_blocks: # of residual blocks in the global generator network.
    :type n_blocks: int
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
    :param norm: normalization.
    :type norm: str
    :param max_gnorm: max grad norm to which it will be clipped (if exceeded).
    :type max_gnorm: float
    :param monitor_gnorm: flag set to monitor grad norms.
    :type monitor_gnorm: bool
    :param save_epoch_freq: frequency of saving checkpoints at the end of epochs.
    :type save_epoch_freq: int
    :param print_freq: frequency of showing training results on console.
    :type print_freq: int
    :param tensorbard: if tensorboard.
    :type tensorbard: bool
    :param tensorbard_writer: tensorboard writer.
    :type tensorbard_writer: tensorbard_writer
    :param testing: if test phase.
    :type testing: bool
    """

    def __init__(self, expr_dir, seed=None, gpu_ids='-1', batch_size=None,
                 epoch_count=1, niter=100, niter_decay=100, beta1=0.5, lr=0.0002,
                 ngf=64, n_blocks=1, input_nc=1, output_nc=1, use_dropout=False, norm='batch', max_gnorm=500.,
                 monitor_gnorm=True, save_epoch_freq=5, print_freq=100, display_freq=100, tensorbard=True,
                 tensorbard_writer=None, testing=False):

        self.expr_dir = expr_dir
        self.seed = seed
        self.gpu_ids = parse_gpu_ids(gpu_ids)
        self.batch_size = batch_size

        self.epoch_count = epoch_count
        self.niter = niter
        self.niter_decay = niter_decay
        self.beta1 = beta1
        self.lr = lr
        self.old_lr = self.lr

        self.ngf = ngf
        self.n_blocks = n_blocks
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.use_dropout = use_dropout
        self.norm = norm
        self.max_gnorm = max_gnorm
        self.w_lambda = torch.tensor(1, dtype=torch.float)

        self.monitor_gnorm = monitor_gnorm
        self.save_epoch_freq = save_epoch_freq
        self.print_freq = print_freq
        self.display_freq = display_freq
        self.tensorbard = tensorbard
        self.tensorbard_writer = tensorbard_writer
        if self.tensorbard:
            self.tensorbard_writer = SummaryWriter(os.path.join(expr_dir, 'TensorBoard'))
        # Set gpu ids
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])

        # define network we need here
        self.netG = networks.define_generator(input_nc=self.input_nc, output_nc=self.output_nc, ngf=self.ngf,
                                              n_blocks=self.n_blocks, use_dropout=self.use_dropout,
                                              gpu_ids=self.gpu_ids)

        # define all optimizers here
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=self.lr, betas=(self.beta1, 0.999))

        # define criterion
        self.criterion = binary_cross_entropy

        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        if not os.path.exists(os.path.join(expr_dir, 'TensorBoard')) and self.tensorbard:
            os.makedirs(os.path.join(expr_dir, 'TensorBoard'))

        if not testing:
            num_params = 0
            with open("%s/nets.txt" % self.expr_dir, 'w') as nets_f:
                num_params += networks.print_network(self.netG, nets_f)
                nets_f.write('# parameters: %d\n' % num_params)
                nets_f.flush()

    def train(self, train_dataset, test_dataset=None):
        """Train the model with a dataset.

        :param train_dataset: training dataset
        :type train_dataset: :class:`DataLoader`
        :param test_dataset: test dataset
        :type test_dataset: :class:`DataLoader`
        """
        self.batch_size = train_dataset.batch_size
        self.save_options()
        out_f = open(f"{self.expr_dir}/results.txt", 'w')
        use_gpu = len(self.gpu_ids) > 0

        if self.seed is not None:
            print(f"using random seed: {self.seed}")
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if use_gpu:
                torch.cuda.manual_seed_all(self.seed)

        total_steps = 0
        print_start_time = time.time()

        for epoch in range(self.epoch_count, self.niter + self.niter_decay + 1):
            epoch_start_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(train_dataset):
                ct = Variable(data['ct'])
                mask = Variable(data['mask'])

                total_steps += self.batch_size
                epoch_iter += self.batch_size

                if use_gpu:
                    ct = ct.cuda()
                    mask = mask.cuda()
                    self.w_lambda = self.w_lambda.cuda()

                if self.monitor_gnorm:
                    losses, visuals, gnorms = self.train_instance(ct, mask)
                else:
                    losses, visuals = self.train_instance(ct, mask)

                if total_steps % self.display_freq == 0:
                    self.visualize_training(visuals, epoch, epoch_iter / self.batch_size)

                if total_steps % self.print_freq == 0:
                    t = (time.time() - print_start_time) / self.batch_size
                    print_log(out_f, format_log(epoch, epoch_iter, losses, t))
                    if self.tensorbard:
                        self.tensorbard_writer.add_scalars('training losses', {'Loss': losses['Loss']}, total_steps)
                    print_start_time = time.time()

            if epoch % self.save_epoch_freq == 0:
                print_log(out_f, 'saving the model at the end of epoch %d, iterations %d' %
                          (epoch, total_steps))
                self.save('latest')
                if test_dataset:
                    train_mae = self.eval_mae(train_dataset)
                    test_mae = self.eval_mae(test_dataset)
                    self.tensorbard_writer.add_scalars('accuracy', {'Train': train_mae,
                                                                    'Test': test_mae}, epoch)

            print_log(out_f, 'End of epoch %d / %d \t Time Taken: %d sec' %
                      (epoch, self.niter + self.niter_decay, time.time() - epoch_start_time))

            if epoch > self.niter:
                self.update_learning_rate()

            out_f.close()
            self.tensorbard_writer.close()

    def train_instance(self, ct, segmentation):
        """Training instance (batch)

        :param ct: input tensor.
        :type ct: Tensor
        :param segmentation: ground truth tensor.
        :type segmentation: Tensor
        :return: losses and visualization data.
        """
        fake_segmentation = self.netG.forward(ct)
        loss = self.w_lambda * self.criterion(fake_segmentation, segmentation)

        self.optimizer_G.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.max_gnorm)
        self.optimizer_G.step()

        losses = OrderedDict([('Loss', loss.data.item())])

        visuals = OrderedDict([('ct', ct.data),
                               ('segmentation_mask', segmentation.data),
                               ('fake_segmentation_mask', fake_segmentation.data)
                               ])
        if self.monitor_gnorm:
            gnorms = OrderedDict([('gnorm', gnorm)])

            return losses, visuals, gnorms

        return losses, visuals

    def synthesize(self, ct):
        """Evaluate a CT image.

        :param ct: CT image.
        :type ct: Tensor
        :return: Synthesized mask.
        :rtype: dict[Tensor]
        """
        fake_segmentation = self.netG.forward(ct)
        visuals = OrderedDict([('fake_segmentation_mask', fake_segmentation.data)])
        return visuals

    def update_learning_rate(self):
        """Update learning rate"""
        lrd = self.lr / self.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, checkpoint_name):
        """Save the model and optimizer.

        :param checkpoint_name: name of the checkpoint.
        :type checkpoint_name: str
        """
        checkpoint_path = os.path.join(self.expr_dir, checkpoint_name)
        checkpoint = {
            'netG': self.netG.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path):
        """Loads an object saved with torch.save from a file.

        :param checkpoint_path: path to the checkpoint.
        :type checkpoint_path: str
        """
        checkpoint = torch.load(checkpoint_path)
        self.netG.load_state_dict(checkpoint['netG'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])

    def eval(self):
        """Sets the module in evaluation mode."""
        self.netG.eval()

    def visualize_training(self, visuals, epoch, index):
        """Save training image for visualization.

        :param visuals: images.
        :type visuals: dict
        :param epoch: epoch
        :type epoch: int
        :param index: index
        :type index: float
        """
        visuals['ct'] = (visuals['ct'] + 1.) * 1250.
        size = visuals['ct'].size()

        images = [img.cpu().unsqueeze(1) for img in visuals.values()]
        vis_image = torch.cat(images, dim=1).view(size[0] * len(images), size[1], size[2], size[3])
        save_path = os.path.join(self.expr_dir, "training_visuals")
        save_path = os.path.join(save_path, 'cycle_%02d_%04d.png' % (epoch, index))
        image = vutils.make_grid(vis_image.cpu(), nrow=len(images))
        image = image[0].numpy()
        with open(save_path, 'wb') as f:
            writer = png.Writer(width=image.shape[1], height=image.shape[0], bitdepth=16, greyscale=True)
            array = image.astype(np.uint16)
            array2list = array[:, :].tolist()
            writer.write(f, array2list)

    def save_options(self):
        """Save model options."""
        # ToDo deal with none default type (not working with parse_opt_file()
        options_file = open(f"{self.expr_dir}/options.txt", 'wt')
        print_log(options_file, '------------ Options -------------')
        for k, v in sorted(self.__dict__.items()):
            print_log(options_file, '%s: %s' % (str(k), str(v)))
        print_log(options_file, '-------------- End ----------------')

    def eval_mae(self, dataset, use_gpu=True):
        """Evaluation metric using MAE.

        :param dataset: training dataset
        :type dataset: :class:`DataLoader`
        :param use_gpu: if gpu
        :type use_gpu: bool
        :return: MAE of the dataset
        :rtype: float
        """
        mae = []
        for batch in dataset:
            ct, mask = Variable(batch['ct']), Variable(batch['mask'])
            if use_gpu:
                ct = ct.cuda()
                mask = mask.cuda()

            fake_mask = self.netG.forward(ct)
            mae.append(l1_loss(mask, fake_mask.data).item())
        return np.mean(mae)
