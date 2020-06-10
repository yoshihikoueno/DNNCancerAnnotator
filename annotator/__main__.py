'''
DNNAnnotator: DNN model to predict cancer segmentation
'''

# built-in
import pdb
import os

# external
import dsargparse

# custom
from .runs import __main__


if __name__ == '__main__':
    __main__.main(prog='python3 -m annotator')
