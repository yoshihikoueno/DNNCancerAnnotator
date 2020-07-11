'''
interface for training models
'''

# built-in
import pdb
import os
import argparse

# external
import dsargparse
import yaml

# customs
from .. import engine
from .. import data
from ..utils import dump
from ..utils import load


def evaluate(
    save_path,
    data_path,
):
    '''
    Evaluate a model with specified configs.
    This funciton will first dump the input arguments,
    then train a model, finally dump reults.

    Args:
        save_path: from where to load weights/configs/results
        data_path (list[str]): path to the data root dir
    '''

    config = load.load_config(os.path.join(save_path, 'options.json'))
    ds = data.eval_ds(data_path, **config['data_options']['eval'])
    model = engine.TFKerasModel(config)
    results = model.eval(
        ds,
        save_path=os.path.join(save_path),
    )
    return results
