'''
provide custom metrics
'''

# built-in
import pdb
import os

# external
import tensorflow as tf


def solve_metric(metric_spec):
    '''
    solve metric spec and return metric instance
    '''
    if isinstance(metric_spec, str): return metric_spec
    elif isinstance(metric_spec, dict):
        assert len(metric_spec) == 1
        pass
    else: raise ValueError

    metric_name, metric_options = list(metric_spec.items())[0]
    class_ = getattr(tf.keras.metrics, metric_name)
    assert issubclass(class_, tf.keras.metrics.Metric)
    instance = class_(**metric_options)
    return instance
