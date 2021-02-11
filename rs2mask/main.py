from options.rs2mask_option import Rs2MaskOptions
from rs2mask.dcm2mask import Dataset

if __name__ == '__main__':
    opt = Rs2MaskOptions().parse()
    dataset = Dataset(opt)
    dataset.make_png()
    dataset.sort_dataset(ratio=0.8, structure='external')
