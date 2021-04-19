import os
import shutil
from unittest import TestCase

import numpy as np
import torch
import torchio as tio

from segmentation_rt.util.util import listdir_full_path, save_image, print_log, format_log, get_subjects


class Test(TestCase):

    def tearDown(self):
        shutil.rmtree("saved_images", ignore_errors=True)

    def test_listdir_full_path(self):
        full_path = ['tests/test_data/cheese_png/ct', 'tests/test_data/cheese_png/max.nii']
        self.assertEqual(listdir_full_path('tests/test_data/cheese_png'), full_path)

    def test_save_image(self):
        array = np.arange(90, dtype=np.int16).reshape((3, 3, 10))
        os.makedirs("saved_images", exist_ok=True)
        save_image(array, "saved_images")
        images = os.listdir("saved_images")
        self.assertEqual(len(images), 10)

    def test_save_image_sliced(self):
        array = np.arange(90, dtype=np.int16).reshape((3, 3, 10))
        os.makedirs("saved_images", exist_ok=True)
        save_image(array, "saved_images", start=3, end=7)
        images = os.listdir("saved_images")
        self.assertEqual(len(images), 4)

    def test_print_log(self):
        save_path = "results.txt"
        out_f = open(save_path, 'w')
        print_log(out_f, "hello world")
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_format_log(self):
        errors = {1.0: 3.14}
        message = format_log(1, 20, errors, 22.22)
        self.assertEqual(message, '(epoch: 1, iteration: 20, time: 22.220) 1.0: 3.140 ')

    def test_get_subjects(self):
        ct = tio.ScalarImage(tensor=torch.rand(1, 3, 3, 2))
        structure = tio.LabelMap(tensor=torch.ones((1, 3, 3, 2), dtype=torch.uint8))

        subject_1_dir = "tests/test_data/subjects/subject_1"

        os.makedirs(subject_1_dir, exist_ok=True)
        ct.save(os.path.join(subject_1_dir, "ct.nii"))
        structure.save(os.path.join(subject_1_dir, "structure.nii"))
        transform = tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(1, (1, 99.0))
        ])
        subject_dataset = get_subjects(os.path.dirname(subject_1_dir), structures=["structure"], transform=transform)
        self.assertEqual(len(subject_dataset), 1)
        shutil.rmtree(os.path.dirname(subject_1_dir), ignore_errors=True)