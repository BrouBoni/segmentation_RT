import shutil
from unittest import TestCase

from util.util import *


class Test(TestCase):

    def tearDown(self):
        shutil.rmtree("saved_images", ignore_errors=True)

    def test_listdir_full_path(self):
        full_path = ['tests/data/cheese_png/ct', 'tests/data/cheese_png/masks']
        self.assertEqual(listdir_full_path('tests/data/cheese_png'), full_path)

    def test_save_image(self):
        array = np.arange(90, dtype=np.int16).reshape((3, 3, 10))
        os.makedirs("saved_images", exist_ok=True)
        save_image(array, "saved_images", 3, 7)
        images = os.listdir("saved_images")
        self.assertEqual(len(images), 4)
