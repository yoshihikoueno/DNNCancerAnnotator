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


def tf_weighted_crossentropy(label, pred, weight=None, weight_add=0, weight_mul=1):
    '''
    calculates weighted loss
    '''
    if weight is None:
        positive_rate = tf_get_positive_rate(label)
        weight = 1 / positive_rate if positive_rate > 0.0 else 1.0

    weight = weight_mul * weight + weight_add - 1.0
    with tf.control_dependencies([tf.debugging.assert_greater_equal(weight, 0.0)]):
        weight_mask = label * weight + tf.ones_like(label)

    label = tf.stack([label == 0, label == 1], axis=-1)
    pred = tf.stack([1 - pred, pred], axis=-1)

    bce = tf.keras.losses.BinaryCrossentropy()
    loss = bce(label, pred, sample_weight=weight_mask)
    return loss


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
