'''
provides various functions to calculate loss
'''

# built-in
import os
import sys
import pdb

# external
import tensorflow as tf

# custom


@tf.function
def tf_weighted_crossentropy(label, pred, weight=None, weight_add=0, weight_mul=1):
    '''
    calculates weighted loss
    '''
    if weight is None:
        positive_rate = tf_get_positive_rate(label)
        weight = 1 / positive_rate if positive_rate > 0.0 else 1.0

    weight = weight_mul * weight + weight_add
    with tf.control_dependencies([tf.debugging.assert_greater_equal(weight, 0.0)]):
        weight_mask = label * weight + tf.ones_like(label)

    label = tf.stack([label == 0, label == 1], axis=-1)
    pred = tf.concat([1 - pred, pred], axis=-1)

    strategy = tf.distribute.get_strategy()
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE)
    loss = bce(label, pred, sample_weight=weight_mask)
    num_replicas = strategy.num_replicas_in_sync
    per_replica_batch_size = tf.shape(loss)[0]
    global_batch_size = per_replica_batch_size * num_replicas
    loss /= tf.cast(global_batch_size, loss.dtype)
    return loss


class TFWeightedCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weight=None, weight_add=0.0, weight_mul=1.0):
        self.weight = weight
        self.weight_add = weight_add
        self.weight_mul = weight_mul

        self.config = dict(
            weight=weight, weight_add=weight_add, weight_mul=weight_mul,
        )
        super().__init__(name='weighted_crossentropy')
        return

    @tf.function
    def call(self, y_true, y_pred):
        loss = tf_weighted_crossentropy(y_true, y_pred, **self.config)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config


def tf_get_positive_rate(label):
    max_value = tf.reduce_max(label)
    min_value = tf.reduce_min(label)

    upbound = tf.debugging.assert_less_equal(max_value, 1.0)
    lowbound = tf.debugging.assert_greater_equal(min_value, 0.0)

    with tf.control_dependencies([upbound, lowbound]):
        positive_rate = tf.reduce_sum(label) / tf.cast(tf.reduce_prod(tf.shape(label)), tf.float32)

    assert_range = [tf.debugging.assert_greater_equal(positive_rate, 0.0), tf.debugging.assert_less_equal(positive_rate, 1.0)]
    with tf.control_dependencies(assert_range):
        return positive_rate


tf.keras.utils.get_custom_objects().update(weighted_crossentropy=tf_weighted_crossentropy)
tf.keras.utils.get_custom_objects().update(WeightedCrossentropy=TFWeightedCrossentropy)
