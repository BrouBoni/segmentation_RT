import shutil
from unittest import TestCase
import numpy as np
import os
from segmentation_rt.util.util import listdir_full_path, save_image


class Test(TestCase):

    def tearDown(self):
        shutil.rmtree("saved_images", ignore_errors=True)

    def test_listdir_full_path(self):
        full_path = ['tests/test_data/cheese_png/ct', 'tests/test_data/cheese_png/max.nii']
        self.assertEqual(listdir_full_path('tests/test_data/cheese_png'), full_path)

    def test_save_image(self):
        array = np.arange(90, dtype=np.int16).reshape((3, 3, 10))
        os.makedirs("saved_images", exist_ok=True)
        save_image(array, "saved_images", start=3, end=7)
        images = os.listdir("saved_images")
        self.assertEqual(len(images), 4)
