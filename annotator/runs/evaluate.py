'''
interface for evaluating models
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
    config,
    save_path,
    data_path,
    tag,
    avoid_overwrite=False,
    prediction_threshold=None,
):
    '''
    Evaluate a model with specified configs
    for every checkpoints available.

    Args:
        config: configuration file path
        save_path: where to find weights/configs/results
        data_path (list[str]): path to the data root dir
        tag: save tag
        avoid_overwrite (bool): should `save_path` altered when a directory already
            exists at the original `save_path` to avoid overwriting.
        prediction_threshold (float): threshold to apply onto the predicted segmentation mask
            default(None): disable threshold
    '''
    config = load.load_config(config)
    ds = data.eval_ds(data_path, **config['data_options']['eval'])
    viz_ds = data.eval_ds(data_path, **config['data_options']['eval'], include_meta=True)

    model = engine.TFKerasModel(config)
    results = model.eval(
        ds, viz_ds,
        tag=tag,
        save_path=os.path.join(save_path),
        avoid_overwrite=avoid_overwrite,
    )
    return results
