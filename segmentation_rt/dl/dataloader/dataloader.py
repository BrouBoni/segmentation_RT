import os

import torch
import torch.utils.data
import torchio as tio

from util.util import get_subjects


class DatasetSingle:
    """Initialize a dataset suited inference.

            :param str root:
                root folder.

            :param list[str] structures:
                list of structures.

            :param (int, int, int) patch_size:
                Tuple of integers (width, height, depth).
    """

    def __init__(self, root, structures, patch_size=(512, 512, 6)):
        self.root = root
        self.structures = structures
        self.n_structures = len(structures)

        self.transform = tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(1, (1, 99.0))
        ])
        self.subject = tio.Subject(ct=tio.ScalarImage(os.path.join(root, 'ct.nii')))

        self.patch_size = patch_size
        self.patch_overlap = 4
        grid_sampler = tio.inference.GridSampler(
            self.transform(self.subject),
            self.patch_size,
            self.patch_overlap,
        )

        self.loader = torch.utils.data.DataLoader(
            grid_sampler, batch_size=4, drop_last=True)
        self.aggregator = tio.inference.GridAggregator(grid_sampler)


class DatasetPatch:
    """Initialize a dataset suited for patch-based training.

        :param str root:
            root folder.

        :param list[str] structures:
            list of structures..

        :param float ratio:
            splitting ratio.

        :param (int, int, int) patch_size:
            Tuple of integers (width, height, depth).

        :param int batch_size:
            batch size.

        :param int num_worker:
            number of subprocesses to use for data loading..

        :param int samples_per_volume:
            number of patches to extract from each volume.

        :param int max_length:
            maximum number of patches that can be stored in the queue.
        """

    def __init__(self, root, structures, ratio=0.9, patch_size=(256, 256, 6),
                 batch_size=1, num_worker=2, samples_per_volume=20, max_length=200):

        self.root = root
        self.structures = structures
        self.n_structures = len(structures)

        self.batch_size = batch_size
        self.num_worker = num_worker

        self.transform = tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(1, (1, 99.0))
        ])

        self.subjects = get_subjects(self.root, self.structures, self.transform)
        self.training_subjects, self.validation_subjects = random_split(self.subjects, ratio)
        self.patches_training_set, self.patches_validation_set = queuing(self.training_subjects,
                                                                         self.validation_subjects,
                                                                         patch_size, samples_per_volume,
                                                                         max_length, num_worker)

    def get_loaders(self):
        """Return training and validation :class:`data.DataLoader`.

        :return: training and validation DataLoader.
        :rtype: (:class:`data.DataLoader`,:class:`data.DataLoader`)
        """
        training_loader_patches = torch.utils.data.DataLoader(
            self.patches_training_set, batch_size=self.batch_size,
            drop_last=True)

        validation_loader_patches = torch.utils.data.DataLoader(
            self.patches_validation_set, batch_size=self.batch_size,
            drop_last=True)

        print('Training set:', len(self.training_subjects), 'subjects')
        print('Validation set:', len(self.validation_subjects), 'subjects')

        return training_loader_patches, validation_loader_patches


def random_split(subjects, ratio=0.8):
    """Randomly split a dataset into non-overlapping new datasets according to the ratio.

    :param subjects: dataset to be split.
    :type subjects: :class:`tio.SubjectsDataset`
    :param ratio: splitting ratio.
    :type ratio: float
    :return: training and validation datasets.
    :rtype: (:class:`tio.SubjectsDataset`, :class:`tio.SubjectsDataset`)
    """
    num_subjects = len(subjects)
    num_training_subjects = int(ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    return torch.utils.data.random_split(subjects, num_split_subjects)


def queuing(training_subjects, validation_subjects, patch_size, samples_per_volume=10,
            max_length=200, num_workers=2):
    """Queue used for stochastic patch-based training.
    See :class:`tio.data.Queue`.

    :param training_subjects: train dataset.
    :type training_subjects: :class:`tio.SubjectsDataset`
    :param validation_subjects: validation dataset.
    :type validation_subjects: :class:`tio.SubjectsDataset`
    :param patch_size: Tuple of integers (width, height, depth).
    :type patch_size: (int, int, int)
    :param samples_per_volume: number of patches to extract from each volume.
    :type samples_per_volume: int
    :param max_length: maximum number of patches that can be stored in the queue.
    :type max_length: int
    :param num_workers: number of subprocesses to use for data loading.
    :type num_workers: int
    :return: training and validation queue.
    :rtype: (:class:`tio.data.Queue`, :class:`tio.data.Queue`)
    """
    sampler = tio.data.WeightedSampler(patch_size, 'label_map')

    patches_training_set = tio.Queue(
        subjects_dataset=training_subjects,
        max_length=max_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_validation_set = tio.Queue(
        subjects_dataset=validation_subjects,
        max_length=max_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=False,
        shuffle_patches=False,
    )

    return patches_training_set, patches_validation_set
