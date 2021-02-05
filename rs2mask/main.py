import os

from rs2mask.rtstruct import RTStruct

if __name__ == '__main__':
    data = os.path.join('rs2mask', 'data', '314159')
    struct = RTStruct(data)
    struct.create()
    struct.save()
    print(struct.ds_rs)
