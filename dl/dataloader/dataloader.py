import os
import torchio as tio
import torch
import torch.utils.data as data


class DatasetSingle:
    def __init__(self, root, structures, patch_size=(512, 512, 16)):
        self.root = root
        self.structures = structures
        self.n_structures = len(structures)

        self.transform = tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(1, (1, 99))
        ])

        self.subjects = get_subjects(self.root, self.structures, self.transform)

        self.patch_size = patch_size
        patch_overlap = 2
        grid_sampler = tio.inference.GridSampler(
            self.subjects[0],
            self.patch_size,
            patch_overlap,
        )

        self.loader = torch.utils.data.DataLoader(
            grid_sampler, batch_size=1, num_workers=0, drop_last=True)
        self.aggregator = tio.inference.GridAggregator(grid_sampler)


class DatasetPatch:
    def __init__(self, root, structures, ratio=0.9, crop_size=(256, 256, 6),
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
                                                                         crop_size, samples_per_volume,
                                                                         max_length, num_worker)

    def get_loaders(self):
        training_loader_patches = torch.utils.data.DataLoader(
            self.patches_training_set, batch_size=self.batch_size,
            drop_last=True)

        validation_loader_patches = torch.utils.data.DataLoader(
            self.patches_validation_set, batch_size=self.batch_size,
            drop_last=True)

        print('Training set:', len(self.training_subjects), 'subjects')
        print('Validation set:', len(self.validation_subjects), 'subjects')

        return training_loader_patches, validation_loader_patches


def get_subjects(path, structures, transform):
    subject_ids = os.listdir(path)
    subjects = []
    for subject_id in subject_ids:
        ct_path = os.path.join(path, subject_id, 'ct.nii')
        structures_path_dict = {k: os.path.join(path, subject_id, k + '.nii') for k in structures}

        subject = tio.Subject(
            ct=tio.ScalarImage(ct_path),
        )
        label_map = torch.zeros(subject["ct"].shape, dtype=torch.long)
        for i, (k, v) in enumerate(structures_path_dict.items()):
            label_map += tio.LabelMap(v).data * (i + 1)

        label_map[label_map > len(structures)] = 0
        subject.add_image(tio.LabelMap(tensor=label_map, affine=subject["ct"].affine), 'label_map')
        subjects.append(subject)

    return tio.SubjectsDataset(subjects, transform=transform)


def random_split(subjects, ratio):
    num_subjects = len(subjects)
    num_training_subjects = int(ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    return torch.utils.data.random_split(subjects, num_split_subjects)


def queuing(training_subjects, validation_subjects, crop_size, samples_per_volume=10,
            max_length=200, num_workers=2):

    sampler = tio.data.WeightedSampler(crop_size, 'label_map')

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
