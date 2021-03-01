import os

from dl.dataloader.dataloader import DataLoader
from dl.model.model import Model
# from mask2rs.rtstruct import RTStruct
from rs2mask.dcm2mask import Dataset

if __name__ == '__main__':

    # dataset
    mask_name = "Parotide D"
    dataset = Dataset('data/XL', 'XL_data', [mask_name])
    # # dataset.make_png()
    # dataset.sort_dataset(ratio=0.9, export_path='datasets', structure=mask_name)

    # training
    # root_training = 'datasets/XL_data/'
    # checkpoints_dir = 'checkpoints/'
    # name = 'seg_trachee'
    # expr_dir = os.path.join(checkpoints_dir, name)
    # train_data_loader = DataLoader(root_training, mask_name, subset='training', batch_size=4, crop_size=256,
    #                                drop_last=True, num_workers=2)
    # train_dataset = train_data_loader.load_data()
    # test_data_loader = DataLoader(root_training, mask_name, subset='testing', batch_size=1, drop_last=True,
    #                               num_workers=2)
    # test_dataset = test_data_loader.load_data()
    # model = Model(expr_dir, n_blocks=9, niter=100, niter_decay=100)
    # model.train(train_dataset, test_dataset)

    # testing
    # expr_dir = os.path.join(checkpoints_dir, name)
    # model = Model(expr_dir, n_blocks=1)
    # root_prediction = 'prediction/ct/'
    # pred_data_loader = DataLoader(root_prediction, mask_name, subset='prediction', batch_size=1, drop_last=True)
    # pred_dataset = pred_data_loader.load_data()
    # model.test(pred_dataset)

    # rtstruct
    # data = os.path.join('prediction')
    # struct = RTStruct(data)
    # struct.create()
    # struct.save()
