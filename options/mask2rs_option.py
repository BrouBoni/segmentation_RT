import argparse


class Mask2RsOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def initialize(self):
        self.parser.add_argument('-s', '--structures', nargs='+', help='Structures to import',
                                 default=['External', 'max', 'aux']
                                 )
        self.parser.add_argument('-r', '--root', type=str, help='path to data',
                                 default='rs2mask/data/cheese'
                                 )
        self.parser.add_argument('-n', '--name', type=str, help='dataset name',
                                 default='dataset_cheese'
                                 )
        self.parser.add_argument('-e', '--export', type=str, help='path to data',
                                 default='datasets'
                                 )
        self.parser.add_argument('-v', '--verbose', type=bool, help='log file',
                                 default='1'
                                 )

    def parse(self):
        if not self.opt:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
