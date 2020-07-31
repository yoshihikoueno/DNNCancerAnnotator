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
    instance = tf.keras.metrics.get({
        "class_name": metric_name,
        "config": metric_options,
    })
    return instance

class FBetaScore(tf.keras.metrics.Metric):
    def __init__(self, beta, epsilon=1e-07, **kargs):
        super().__init__(**kargs)
        assert beta > 0
        self.beta = beta
        self.epsilon = epsilon
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)
        return

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        score = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + self.epsilon)
        return score

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
        return

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = super().get_config()
        config.update({'beta': self.beta, 'epsilon': self.epsilon})
        return config


tf.keras.utils.get_custom_objects().update(FBetaScore=FBetaScore)
