from rs2mask.dcm2mask import Dataset

if __name__ == '__main__':
    dataset = Dataset('rs2mask/data/cheese', 'dataset_cheese', ['External', 'max', 'aux'], 'datasets')
    dataset.make_png()
    dataset.sort_dataset(ratio=0.8, structure='external')
