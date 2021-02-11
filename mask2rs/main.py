import os

from mask2rs.rtstruct import RTStruct

if __name__ == '__main__':
    data = os.path.join('mask2rs', 'data', 'cheese_png')
    struct = RTStruct(data)
    struct.create()
    struct.save()
    print(struct.ds_rs)
