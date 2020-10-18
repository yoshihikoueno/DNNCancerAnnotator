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


def train(
    config,
    save_path,
    data_path,
    max_steps,
    early_stop_steps=None,
    save_freq=500,
    validate=False,
    val_data_path=None,
    visualize=False,
    profile=False,
):
    '''
    Train a model with specified configs.
    This funciton will first dump the input arguments,
    then train a model, finally dump reults.

    Args:
        config (list[str]): configuration file path
            This option accepts arbitrary number of configs.
            If a list is specified, the first one is considered
            as a "main" config, and the other ones will overwrite the content
        save_path: where to save weights/configs/results
        data_path (list[str]): path to the data root dir
        max_steps (int): max training steps
        early_stop_steps: steps to train without improvements
            None(default) disables this feature
        save_freq: interval of checkpoints
            default: 500 steps
        validate: also validate the model on the validation dataset
        val_data_path (list[str]): path to the validation dataset
        visualize (bool): should visualize results
        profile (bool): enable profilling
    '''
    config = load.load_config(config)
    dump.dump_options(
        os.path.join(save_path, 'options.yaml'),
        avoid_overwrite=True,
        config=config,
        save_path=save_path,
        data_path=data_path,
    )

    ds = data.train_ds(data_path, **config['data_options']['train'])
    if validate:
        assert val_data_path is not None
        val_ds = data.eval_ds(val_data_path, **config['data_options']['eval'])
    else: val_ds = None

    if visualize:
        visualization = {
            'train': data.eval_ds(data_path, **config['data_options']['eval'], include_meta=True),
            'validation': data.eval_ds(val_data_path, **config['data_options']['eval'], include_meta=True),
        }
    else: visualization = {}

    model = engine.TFKerasModel(config)
    results = model.train(
        ds,
        save_path=os.path.join(save_path),
        max_steps=max_steps,
        early_stop_steps=early_stop_steps,
        save_freq=save_freq,
        val_data=val_ds,
        visualization=visualization,
        profile=profile,
    )

    dump.dump_train_results(
        os.path.join(save_path, 'results.yaml'),
        results,
        format_='yaml',
    )
    return results
