import os

from dataloader.data_loader import DataLoader
from model.model import Model

if __name__ == "__main__":
    root_training = 'datasets/dataset_parotide/'
    root_prediction = 'datasets/314/'

    checkpoints_dir = 'checkpoints/'
    mask_name = 'parotide_d'
    name = 'seg3'
    expr_dir = os.path.join(checkpoints_dir, name)
    # train_data_loader = DataLoader(root_training, mask_name, subset='training', crop_size=256)
    # train_dataset = train_data_loader.load_data()
    # test_data_loader = DataLoader(root_training, mask_name, subset='testing', batch_size=1, drop_last=True)
    # test_dataset = test_data_loader.load_data()
    model = Model(expr_dir)
    # model.train(train_dataset, test_dataset)
    pred_data_loader = DataLoader(root_prediction, mask_name, subset='prediction', batch_size=1, drop_last=True)
    pred_dataset = pred_data_loader.load_data()

    model.test(pred_dataset)
