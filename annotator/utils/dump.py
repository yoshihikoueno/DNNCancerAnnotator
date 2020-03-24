'''
provide vairous functions to dump data
'''

# built-in
import pickle
import json
import os

# external

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
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)

    if format_ == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(train_results, f)
    else: raise NotImplementedError(f'Umimplemented format {format_}')
    return
