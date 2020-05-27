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
    model_config,
    save_path,
    data_path,
    max_steps,
    early_stop_steps,
):
    '''
    Train a model with specified configs.
    This funciton will first dump the input arguments,
    then train a model, finally dump reults.

    Args:
        model_config: model configuration
            this can be a dict contianing configs,
            or a string path to the config file
        save_path: where to save weights/configs/results
        data_path: path to the data root dir
    '''
    if isinstance(model_config, str):
        with open(model_config) as f:
            model_config = yaml.safe_load(f)
    assert isinstance(model_config, dict)

    dump.dump_options(
        os.path.join(save_path, 'options.json'),
        format_='json',
        model_config=model_config,
        save_path=save_path,
        data_path=data_path,
    )

    ds = data.train_ds(data_path)
    model_config = load.load_model_config(model_config)
    model = engine.TFKerasModel(model_config, save_path)
    results = model.train(ds)

    dump.dump_train_results(
        os.path.join(save_path, 'resutls.pickle'),
        results,
    )
    return
