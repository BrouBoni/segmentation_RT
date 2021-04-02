import os

from dl.dataloader.dataloader import DatasetPatch, DatasetSingle
from dl.model.model import Model
from mask2rs.rtstruct import RTStruct
from rs2mask.dcm2mask import Dataset

if __name__ == '__main__':

    # dataset
    structures = ["Coeur", "Sein G", "Sein D"]
    dataset = Dataset('data/data', 'data/DIBH_dataset', structures)
    dataset.make()

    # training
    root_training = 'data/DIBH_dataset/'
    checkpoints_dir = 'checkpoints/'
    name = 'DIBH'

    expr_dir = os.path.join(checkpoints_dir, name)
    dataset = DatasetPatch(root_training, structures, 0.9, batch_size=4)
    training_loader_patches, validation_loader_patches = dataset.get_loaders()
    model = Model(expr_dir, structures, n_blocks=9, niter=150, niter_decay=50)
    model.train(training_loader_patches, validation_loader_patches)

    # testing
    expr_dir = os.path.join(checkpoints_dir, name)
    model = Model(expr_dir, structures,  n_blocks=9)
    root_prediction = 'prediction/'
    pred_data_loader = DatasetSingle(root_prediction, structures)
    fake_segmentation = model.test(pred_data_loader, export_path='prediction/143012/', save=True)

    # rtstruct
    ct_path = os.path.join('prediction/143012/ct/')
    mask = os.path.join('prediction/143012/nii/fake_segmentation.nii')

    struct = RTStruct(ct_path, mask, structures)
    struct.create()
    struct.save()
