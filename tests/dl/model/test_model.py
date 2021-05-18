import os.path
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from segmentation_rt.dl.model.model import *

torch.manual_seed(0)


class TestModel(TestCase):

    def setUp(self):
        self.root_training = 'tests/test_data/cheese_dcm'
        self.checkpoints_dir = 'tests/test_data/checkpoints/'
        self.name = 'test_model'
        self.expr_dir = os.path.join(self.checkpoints_dir, self.name)

        self.structures = ['max']
        self.model = Model(self.expr_dir, self.structures, n_blocks=1, niter=1, niter_decay=1)

    def tearDown(self):
        shutil.rmtree(self.checkpoints_dir, ignore_errors=True)

    def test_update_learning_rate(self):
        self.assertEqual(self.model.lr, 0.0002)
        self.model.update_learning_rate()
        self.assertEqual(self.model.old_lr, 0.0)

    def test_save(self):
        self.model.save("latest")
        self.assertTrue(os.path.exists(os.path.join(self.expr_dir, "latest")))

