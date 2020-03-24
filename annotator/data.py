'''
this module will provide abstraction layer for the dataset API
'''

# built in

# external
import tensorflow as tf

# customs


def train_ds(path):
    '''
    generate dataset for training
    '''
    ds = base(path)
    ds = augment(ds)
    return ds

def eval_ds(path):
    '''
    generate dataset for evaluation
    '''
    ds = base(path)
    return ds

def predict_ds(path):
    '''
    generate dataset for prediction
    '''
    ds = base(path)
    return ds

def base(path):
    return ds

def augment(ds):
    return ds
