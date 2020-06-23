'''
interface for training models
'''

# built-in
import pdb
import os
import argparse
import pdb

# external
import dsargparse
import yaml

# customs
from .. import engine
from .. import data
from ..utils import dump
from ..utils import load


def train(
    config,
    save_path,
    data_path,
    max_steps,
    early_stop_steps=None,
    save_freq=500,
):
    '''
    Train a model with specified configs.
    This funciton will first dump the input arguments,
    then train a model, finally dump reults.

    Args:
        config: configuration file path
        save_path: where to save weights/configs/results
        data_path: path to the data root dir
        max_steps (int): max training steps
        early_stop_steps: steps to train without improvements
            None(default) disables this feature
        save_freq: interval of checkpoints
            default: 500 steps
    '''

    dump.dump_options(
        os.path.join(save_path, 'options.json'),
        format_='json',
        config=config,
        save_path=save_path,
        data_path=data_path,
    )

    config = load.load_config(config)
    ds = data.train_ds(data_path, **config['data_options'])
    model = engine.TFKerasModel(config)
    results = model.train(
        ds,
        save_path=os.path.join(save_path),
        max_steps=max_steps,
        early_stop_steps=early_stop_steps,
        save_freq=save_freq,
    )

    dump.dump_train_results(
        os.path.join(save_path, 'resutls.pickle'),
        results,
    )
    return results
