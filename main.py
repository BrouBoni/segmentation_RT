import os

from dl.dataloader.dataloader import DatasetPatch
from dl.model.model import Model
# from mask2rs.rtstruct import RTStruct
from rs2mask.dcm2mask import Dataset

if __name__ == '__main__':

    # dataset
    structures = ["Parotide D", "Parotide G", "Trachee"]
    # dataset = Dataset('data/XL', 'data/XL_3D', structures)
    # dataset.make()
    # sort_dataset('data/XL_3D', 'datasets/XL_3D_dataset', structures, 0.9)

    # training
    root_training = 'data/XL_3D/'
    checkpoints_dir = 'checkpoints/'
    name = 'seg_3D'
    expr_dir = os.path.join(checkpoints_dir, name)

    dataset = DatasetPatch(root_training, structures, 0.9, batch_size=2, num_worker=2)
    training_loader_patches, validation_loader_patches = dataset.get_loaders()

    model = Model(expr_dir, structures, n_blocks=9, niter=100, niter_decay=100)
    model.train(training_loader_patches, validation_loader_patches)

    # testing
    # expr_dir = os.path.join(checkpoints_dir, name)
    # model = Model(expr_dir, n_blocks=9)
    # root_prediction = 'prediction/ct/'
    # pred_data_loader = DataLoader(root_prediction, structures, subset='prediction', batch_size=1, drop_last=True)
    # pred_dataset = pred_data_loader.load_data()
    # model.test(pred_dataset)

    # rtstruct
    # data = os.path.join('prediction')
    # struct = RTStruct(data)
    # struct.create()
    # struct.save()
