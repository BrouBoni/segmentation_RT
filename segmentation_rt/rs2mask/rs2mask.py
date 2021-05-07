""" Implementation of :py:class:`Dataset` object. A folder containing a set of subjects with CT and RS in dicom format
is converted into nii format. A new folder is created keeping the same organization.
"""

import os

import numpy as np
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs


class Dataset:
    """
    From dicom to dataset class. Convert CT and RTSTRUCT into nii, readable by deep learning frameworks.

    All subfolders representing subject must contain the CT and the RS associated.

    Example:
        >>> from segmentation_rt.rs2mask import Dataset
        >>> structures = ['Heart', 'Breast L', 'Breast R']
        >>> dataset = Dataset('data/dicom_dataset', 'data/nii_dataset', structures)
        >>> dataset.make()

    :param string path:
        Root directory.

    :param string export_path:
        Export path.

    :param list[string] structures:
        List of desired structure(s).

    :param bool force:
        Force export even if one structure is missing.
    """

    def __init__(self, path, export_path, structures, force=True):
        self.path = path
        self.export_path = export_path
        self.structures = structures
        self.dataset_name = os.path.basename(export_path)
        self.force = force

        self.root_path = os.path.dirname(self.path)
        self.patients = [folder for folder in os.listdir(self.path) if
                         os.path.isdir(os.path.join(self.path, folder))]
        self.patient_paths = [os.path.join(self.path, patient) for patient in self.patients]
        self.rs_paths = self.get_rs()

    def __str__(self):
        return self.dataset_name

    def get_rs(self):
        """
        List RTSTRUCT for each patient.

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
        List missing and not missing structures in a RTSTRUCT.

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
            print(f"WARNING ! Some structures are missing:  {missing}\n")

        return missing, not_missing

    def make(self):
        """Create structures and convert the CT in nii format for each subject."""
        print(f"Structure(s) to export: {self.structures}")
        print(f"Patient(s) identification : {self.patients}")

        for index, path_patient in enumerate(self.patient_paths):
            patient_id = self.patients[index]
            print(f"Exporting {index + 1} ({patient_id}) on {len(self.patients)}")

            nii_output = os.path.join(self.export_path, patient_id)
            missing, not_missing = self.find_structures(index)
            if len(missing) == 0 or self.force:

                dcmrtstruct2nii(self.rs_paths[index], path_patient, nii_output, not_missing, False,
                                mask_foreground_value=1)

                nii_maks = [nii_mask for nii_mask in os.listdir(nii_output) if nii_mask.startswith('mask')]
                for nii in nii_maks:
                    name = os.path.splitext(nii)[0].split("_")[1].replace("-", " ")
                    os.rename(os.path.join(nii_output, nii), os.path.join(nii_output, name + '.nii'))

                os.rename(os.path.join(nii_output, "image.nii"), os.path.join(nii_output, "ct.nii"))
            else:
                print(f"Skip {patient_id} because of missing structure(s)")

        print("Export done")
