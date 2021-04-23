from segmentation_rt.dl.dataloader.dataloader import DatasetPatch
from tests.dl.utils import TorchioTestCase


class TestDatasetPatch(TorchioTestCase):

    def setUp(self):
        super().setUp()
        self.dataset_patch = DatasetPatch(self.dir, ["label"], patch_size=(5, 5, 5),
                                          num_worker=0, samples_per_volume=2, max_length=4)

    def test_get_loaders(self):
        training_loader_patches, validation_loader_patches = self.dataset_patch.get_loaders()
        self.assertEqual(len(training_loader_patches), 4)
        self.assertEqual(len(validation_loader_patches), 2)

