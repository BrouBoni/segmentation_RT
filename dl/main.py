import os

from dataloader.data_loader import DataLoader
from model.model import Model

if __name__ == "__main__":
    root = 'datasets/dataset_parotide/'
    checkpoints_dir = 'checkpoints/'
    mask_name = 'parotide_d'
    name = 'seg3'
    expr_dir = os.path.join(checkpoints_dir, name)
    train_data_loader = DataLoader(root, mask_name, subset='training', crop_size=256)
    train_dataset = train_data_loader.load_data()
    model = Model(expr_dir)
    model.train(train_dataset)
