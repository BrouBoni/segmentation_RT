import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torchio as tio


class TorchioTestCase(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        self.dir = Path(tempfile.gettempdir()) / '.segmentation_rt_tests'
        self.dir.mkdir(exist_ok=True)
        random.seed(42)
        np.random.seed(42)

        patients = ["a", "b", "c"]

        for patient in patients:
            patient_dir = os.path.join(self.dir, patient)
            os.mkdir(patient_dir)
            self.get_image_path(os.path.join(patient_dir, "ct"))
            self.get_image_path(os.path.join(patient_dir, "label"), binary=True)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        shutil.rmtree(self.dir)

    def get_image_path(
            self,
            stem,
            binary=False,
            shape=(10, 10, 20),
            spacing=(1, 1, 1),
            components=1,
            force_binary_foreground=True,
            ):
        shape = (*shape, 1) if len(shape) == 2 else shape
        data = np.random.rand(components, *shape)
        if binary:
            data = (data > 0.5).astype(np.uint8)
            if not data.sum() and force_binary_foreground:
                data[..., 0] = 1
        else:
            data *= 100
            data = data.astype(np.uint8)

        affine = np.diag((*spacing, 1))

        path = self.dir / f'{stem}{".nii"}'
        path = str(path)
        image = tio.ScalarImage(
            tensor=data,
            affine=affine,
        )
        image.save(path)

