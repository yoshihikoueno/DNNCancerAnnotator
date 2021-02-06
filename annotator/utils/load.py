'''
provide various functions to load data.
'''

# built-in
import os
import json
import pickle

# external
import yaml

# customs


def load_config(path):
    '''
    Load configs.
    This function can load different types of formats.
    The file format is determined by its extension.

    Args:
        path: path to the config file
            can be a single config file (str)
            or a list of config files (list[str]).
            If a list is specified, the first one is considered
            as a "main" config, and the other ones will overwrite the content
            of the main config.

    Returns:
        config (whatever dumped in a file)
    '''
    if isinstance(path, str): return load_config([path])
    assert isinstance(path, (tuple, list))
    assert path

    configs = list(map(_load_config_single, path))
    config = configs[0]
    for additional_conf in configs[1:]:
        config = _apply_config(config, additional_conf)
    return config


def _apply_config(base_config, add_config):
    '''update the content of base_config with add_config'''
    def _apply(target, dest, value):
        if '.' not in dest:
            target[dest] = value
        else:
            keys = dest.split('.')
            if keys[0] not in target: target[keys[0]] = dict()
            _apply(target[keys[0]], '.'.join(keys[1:]), value)
        return target

    for key, val in add_config.items():
        base_config = _apply(base_config, key, val)
    return base_config


def _load_config_single(path):
    '''
    Load configs.
    This function can load different types of formats.
    The file format is determined by its extension.

    Args:
        path: path to the config file

    Returns:
        config (whatever dumped in a file)
    '''
    extension = os.path.splitext(path)[1][1:]

    if extension == 'json':
        with open(path) as f:
            config = json.load(f)
    elif extension == 'yaml':
        with open(path) as f:
            config = yaml.safe_load(f)
    elif extension == 'pickle':
        with open(path, 'rb') as f:
            config = pickle.load(f)
    else: raise NotImplementedError(f'Unexpected extension {extension}')
    return config
