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
    save_path,
    data_path,
    tag,
    config=None,
    avoid_overwrite=False,
    export_path=None,
    export_images=False,
    export_csv=False,
    visualize_sensitivity=False,
    min_interval=1,
    step_range=None,
    overlay=False,
    skip_visualization=False,
):
    '''
    Evaluate a model with specified configs
    for every checkpoints available.

    Args:
        save_path: where to find weights/configs/results
        data_path (list[str]): path to the data root dir
        tag: save tag
        config (str): configuration file path
            None (default): load config from save_path
        avoid_overwrite (bool): should `save_path` altered when a directory already
            exists at the original `save_path` to avoid overwriting.
        export_path (str): path to export results
        export_images (bool): export images
        visualize_sensitivity (bool): whether sensitivity should be visualized
        export_csv (bool): export results csv
        min_interval (int): minimum interval in steps between evaluations.
            Checkpoints which are less than `min_interval` steps away
            from the previous one will be disregarded.
        step_range (tuple[int]): range of steps to evaluate.
            Format: "--step_range start end".
            Default: evaluate the checkpoint at all the steps.
        overlay (bool): whether visualized segmentation should be overlayed
            on top of input image.
        skip_visualization (bool): whether the visualization should be skipped.
    '''
    if config is None:
        config = os.path.join(save_path, 'options.yaml')
        config = load.load_config(config)['config']
    else: config = load.load_config(config)
    ds = data.eval_ds(data_path, **config['data_options']['eval'])

    if skip_visualization: viz_ds = None
    else: viz_ds = data.eval_ds(data_path, **config['data_options']['eval'], include_meta=True)

    model = engine.TFKerasModel(config)
    results = model.eval(
        ds, viz_ds=viz_ds,
        tag=tag,
        save_path=os.path.join(save_path),
        avoid_overwrite=avoid_overwrite,
        export_path=export_path,
        export_images=export_images,
        export_csv=export_csv,
        visualize_sensitivity=visualize_sensitivity,
        min_interval=min_interval,
        step_range=step_range,
        overlay=overlay,
    )
    return results
