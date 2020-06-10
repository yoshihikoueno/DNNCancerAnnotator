'''
DNNAnnotator: CLI interface
'''

# built-in
import pdb
import os
import argparse

# external
import dsargparse

# customs
from .. import engine
from .. import data
from . import train, evaluate, predict
from ..utils import dump
from ..utils import load


def main(prog='python3 -m annotator.runs'):
    parser = dsargparse.ArgumentParser(main=main, prog=prog)
    subparsers = parser.add_subparsers(help='command')
    subparsers.add_parser(train.train, add_arguments_auto=True)
    return parser.parse_and_run()


if __name__ == '__main__':
    main()
