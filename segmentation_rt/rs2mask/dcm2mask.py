import os

import numpy as np
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs


class Dataset:
    """
    From dicom to dataset class. Convert CT and RTs files into PNG, readable by deep learning frameworks.

    :param string path:
        Root directory.

    :param string export_path:
        Export path.

    :param List[string] structures:
        List of desired structure(s).
    """

    def __init__(self, path, export_path, structures):
        self.path = path
        self.export_path = export_path
        self.structures = structures
        self.dataset_name = os.path.basename(export_path)

        self.root_path = os.path.dirname(self.path)
        self.patients = [folder for folder in os.listdir(self.path) if
                         os.path.isdir(os.path.join(self.path, folder))]
        self.patient_paths = [os.path.join(self.path, patient) for patient in self.patients]
        self.rs_paths = self.get_rs()

        self.path_dataset = os.path.join(self.root_path, self.dataset_name)

    def __str__(self):
        return self.dataset_name

    def get_rs(self):
        """
        List RT Structure file for each patient.

        :rtype: list[str]
        """
        rs_paths = []
        for path in self.patient_paths:
            files = [filename for filename in os.listdir(path) if filename.startswith("RS")]
            assert len(files) > 0, 'at least one RS is required'
            rs = files[0]
            rs_paths.append(os.path.join(path, rs))
        return rs_paths

    def find_structures(self, index):
        """
        List missing and not missing structures in a RT Structure file.

        :param index: index of the patient.
        :type index: int
        :return: List missing and not missing structures.
        :rtype: (list[str],list[str])
        """
        structures = list_rt_structs(self.rs_paths[index])
        ref_structures = np.array(self.structures)
        maks = np.in1d(ref_structures, structures)
        not_missing = ref_structures[maks]
        missing = ref_structures[~maks]

        if len(missing):
            print(f"WARNING ! Some structures are missing :  {missing}\n")

        return missing, not_missing

    def make(self):
        """Create structures for each structure for all patients."""
        print(f"Structure(s) to export: {self.structures}")
        print(f"Patient(s) identification : {self.patients}")

        for index, path_patient in enumerate(self.patient_paths):
            patient_id = self.patients[index]
            print(f"Exporting {index + 1} ({patient_id}) on {len(self.patients)}")

            nii_output = os.path.join(self.export_path, patient_id)
            _, not_missing = self.find_structures(index)
            dcmrtstruct2nii(self.rs_paths[index], path_patient, nii_output, not_missing, False, mask_foreground_value=1)

            nii_maks = [nii_mask for nii_mask in os.listdir(nii_output) if nii_mask.startswith('mask')]
            for nii in nii_maks:
                name = os.path.splitext(nii)[0].split("_")[1].replace("-", " ")
                os.rename(os.path.join(nii_output, nii), os.path.join(nii_output, name+'.nii'))

            os.rename(os.path.join(nii_output, "image.nii"), os.path.join(nii_output, "ct.nii"))

        print("Export done")
