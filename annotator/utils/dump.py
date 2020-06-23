'''
provide vairous functions to dump data
'''

# built-in
import pickle
import json
import os

# external
import yaml

# custom


def dump_options(path, format_='json', **options):
    '''
    Dump options to file.

    Args:
        path: path to the output file
        format_: output file format
        options: options to be dumped

    Returns:
        None
    '''
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)

    if format_ == 'json':
        with open(path, 'w') as f:
            json.dump(options, f)
    elif format_ == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(options, f)
    else: raise NotImplementedError(f'Umimplemented format {format_}')
    return


def dump_train_results(path, train_results, format_='pickle'):
    '''
    Dumps training results to a file.

    Args:
        path: path to the output file
        train_results: training results to be saved
        format_: output format

    Returns:
        None
    '''
    format_ = format_.lower()
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)

    dump_content = {
        'epoch': train_results.epoch,
        'history': train_results.history,
        'params': train_results.params,
        'model': type(train_results.model).__name__,
    }

    if format_ == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(dump_content, f)
    elif format_ == 'yaml':
        with open(path, 'w') as f:
            yaml.safe_yaml(dump_content, f)
    else: raise NotImplementedError(f'Umimplemented format {format_}')
    return
