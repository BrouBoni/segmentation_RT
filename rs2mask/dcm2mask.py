import os
import random
import shutil
from typing import Union

import nibabel as nib
import numpy as np
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs

from util.util import listdir_full_path, save_image


class Dataset:

    path_dataset: Union[str]

    def __init__(self, opt, path=None, structures=None, name=None):
        self.path = path or opt.root
        self.ref_structures = structures or opt.structures
        self.dataset_name = name or opt.name

        self.root_path = os.path.dirname(self.path)
        self.export_path = os.path.join(opt.export, opt.name)
        self.patients = [folder for folder in os.listdir(self.path) if
                         os.path.isdir(os.path.join(self.path, folder))]
        self.patient_paths = [os.path.join(self.path, patient) for patient in self.patients]
        self.rs_paths = self.get_rs()

        self.path_dataset = os.path.join(self.root_path, self.dataset_name)

    def __str__(self):
        return self.dataset_name

    def get_rs(self):
        rs_paths = []
        for path in self.patient_paths:
            files = [filename for filename in os.listdir(path) if filename.startswith("RS")]
            assert len(files) > 0, 'at least one RS is required'
            rs = files[0]
            rs_paths.append(os.path.join(path, rs))
        return rs_paths

    def find_structures(self, index):
        structures = list_rt_structs(self.rs_paths[index])
        ref_structures = np.array(self.ref_structures)
        maks = np.in1d(ref_structures, structures)
        not_missing = ref_structures[maks]
        missing = ref_structures[~maks]

        if len(missing):
            print(f"\n********************\n-> WARNING ! Some structures are missing :  {missing}\n")

        return missing, not_missing

    def nii_to_png(self, name, nii, patient_id):
        image = nii.get_fdata()
        name = name.lower()
        image = np.fliplr(np.rot90(np.asarray(image), 3))

        if name == 'ct':
            image = image + 1024

        save_path = os.path.join(self.path_dataset, patient_id, name)
        os.makedirs(save_path, exist_ok=True)
        save_image(image, save_path)

    def make_png(self):
        print(f"Structure(s) to export: {self.ref_structures}")
        print(f"Patient(s) identification : {self.patients}")

        for index, path_patient in enumerate(self.patient_paths):
            print(f"Exporting {index + 1} on {len(self.patients)}")
            patient_id = self.patients[index]
            nii_output = os.path.join(path_patient, "output")

            _, not_missing = self.find_structures(index)
            dcmrtstruct2nii(self.rs_paths[index], path_patient, nii_output, not_missing, False, mask_foreground_value=1)

            nii_maks = [nii_mask for nii_mask in os.listdir(nii_output) if nii_mask.startswith('mask')]
            for nii in nii_maks:
                nii_object = nib.load(os.path.join(nii_output, nii))
                name = os.path.splitext(nii)[0].split("_")[1].replace("-", "_")
                self.nii_to_png(name, nii_object, patient_id)

            ct_nii_object = nib.load(os.path.join(nii_output, "image.nii"))
            self.nii_to_png("ct", ct_nii_object, patient_id)

            shutil.rmtree(nii_output)
        print(f"Export done")

    def sort_dataset(self, ratio=0.8, structure=None):
        path_dataset = self.path_dataset
        path_out = self.export_path
        print(f"Making {structure} dataset at {path_out}")

        folders = [structure, 'ct']
        for folder in folders:
            os.makedirs(os.path.join(path_out, "train", folder), exist_ok=True)
            os.makedirs(os.path.join(path_out, "test", folder), exist_ok=True)

        patients = [patient for patient in os.listdir(path_dataset) if not patient.startswith('.')]
        patients = np.array([patient for patient in patients if not os.path.exists(os.path.join(patient, structure))])

        # Setup for random choice
        n_patients = len(patients)
        n_patients_train = round(n_patients * ratio)

        #  random choice of which patient goes to train and which one goes to test
        mask = [False] * (n_patients - n_patients_train) + [True] * n_patients_train
        random.shuffle(mask)
        mask = np.array(mask)

        train_patient = patients[mask]
        test_patient = patients[~mask]

        print(f"train = {train_patient}\ntest = {test_patient} \n")

        for train, patient in zip(mask, patients):
            for folder in folders:
                file_path = os.path.join(path_dataset, patient, folder)
                file_paths = listdir_full_path(file_path)
                file_names = os.listdir(file_path)
                for file_name, file_path in zip(file_names, file_paths):
                    if train:
                        file_destination = os.path.join(path_out, "train", folder, str(file_name))
                    else:
                        file_destination = os.path.join(path_out, "test", folder, str(file_name))

                    shutil.copyfile(file_path, file_destination)
