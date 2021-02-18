import argparse
import os

import torch


def create_sub_dirs(opt, sub_dirs):
    for sub_dir in sub_dirs:
        dir_path = os.path.join(opt.expr_dir, sub_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        setattr(opt, sub_dir, dir_path)


class TrainOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, required=True, help='path to data')
        self.parser.add_argument('--name', type=str, required=True,
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
        self.parser.add_argument('--mask_name', type=str, required=True, help='path to data')
        # data
        self.parser.add_argument('--load_size', type=int, default=512, help='image size')
        self.parser.add_argument('--crop', action='store_true', help='crop image')
        self.parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')

        # exp
        self.parser.add_argument('--seed', type=int, help='manual seed')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--niter', type=int, default=150, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=150,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # model
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--n_blocks', type=int, default=9,
                                 help='number of residual blocks in the global generator network')
        self.parser.add_argument('--norm', type=str, default='batch',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--max_gnorm', type=float, default=500.,
                                 help='max grad norm to which it will be clipped (if exceeded)')

        # monitoring
        self.parser.add_argument('--monitor_gnorm', type=bool, default=True, help='flag set to monitor grad norms')
        self.parser.add_argument('--display_freq', type=int, default=5000,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')

        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_server', type=str, default="http://localhost",
                                 help='visdom server of the web display')
        self.parser.add_argument('--display_env', type=str, default='main',
                                 help='visdom display environment name (default is "main")')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')

        self.initialized = True

    def parse(self, sub_dirs=None):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            gpu_id = int(str_id)
            if gpu_id >= 0:
                self.opt.gpu_ids.append(gpu_id)

        # Set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.opt.expr_dir = expr_dir

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        # create sub dirs
        if sub_dirs is not None:
            create_sub_dirs(self.opt, sub_dirs)

        return self.opt


class TestOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--checkpoint_path', required=True, type=str,
                                 help='path to checkpoint -- we assume expr_dir is containing dir')
        self.parser.add_argument('--res_dir', type=str, default='test_res',
                                 help='results directory (will create under expr_dir)')
        self.parser.add_argument('--train_logvar', type=int, default=1, help='train logvar_B on training data')
        self.parser.add_argument('--dataroot', required=True, type=str, help='path to images')
        self.parser.add_argument('--metric', default='visual', type=str, choices=['bpp', 'mse', 'visual', 'noise_sens'])

    def parse(self):
        return self.parser.parse_args()


def parse_opt_file(opt_path):
    def parse_val(s):
        if s == 'None':
            return None
        if s == 'True':
            return True
        if s == 'False':
            return False
        if s == 'inf':
            return float('inf')
        try:
            f = float(s)
            # special case
            if '.' in s:
                return f
            i = int(f)
            return i if i == f else f
        except ValueError:
            return s

    opt = None
    with open(opt_path) as f:
        opt = dict()
        for line in f:
            if line.startswith('-----'):
                continue
            k, v = line.split(':')
            opt[k.strip()] = parse_val(v.strip())
    return opt
