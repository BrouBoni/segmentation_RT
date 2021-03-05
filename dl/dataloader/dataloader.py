import os
import torchio as tio
import torch
import torch.utils.data as data


class DatasetPatch:
    def __init__(self, root, structures, ratio=0.9, crop_size=(64, 64, 6),
                 batch_size=1, num_worker=0, samples_per_volume=1, max_length=100):
        # 128, 128, 64 20
        self.root = root
        self.structures = structures
        self.n_structures = len(structures)

        self.batch_size = batch_size
        self.num_worker = num_worker

        self.transform = tio.Compose([
                tio.ToCanonical(),
                tio.RescaleIntensity(0, (0.5, 99.5))
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
            self.patches_validation_set, batch_size=1,
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

        for k, v in structures_path_dict.items():
            subject.add_image(tio.LabelMap(v), k)

        subjects.append(subject)

    return tio.SubjectsDataset(subjects, transform=transform)


def random_split(subjects, ratio):
    num_subjects = len(subjects)
    num_training_subjects = int(ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    return torch.utils.data.random_split(subjects, num_split_subjects)


def queuing(training_subjects, validation_subjects, crop_size, samples_per_volume=2,
            max_length=4, num_workers=2):

    sampler = tio.data.UniformSampler(crop_size)

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
