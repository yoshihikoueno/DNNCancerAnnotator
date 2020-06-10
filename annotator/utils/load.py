'''
provide various funcitons to load data.
'''

# built-in
import os
import json
import pickle

# external
import yaml

# customs


def load_model_config(path):
    '''
    Load model configs.
    This function can load different types of formats.
    The file format is determined by its extension.

    Args:
        path: path to the config file

    Returns:
        model config (whatever dumped in a file)
    '''
    extension = os.path.splitext(path)[1][1:]

    if extension == 'json':
        with open(path) as f:
            model_config = json.load(f)
    elif extension == 'yaml':
        with open(path) as f:
            model_config = yaml.safe_load(f)
    elif extension == 'pickle':
        with open(path, 'rb') as f:
            model_config = pickle.load(f)
    else: raise NotImplementedError(f'Unexpected extension {extension}')
    return model_config
