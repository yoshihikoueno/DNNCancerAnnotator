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
def tf_weighted_crossentropy(label, pred, weight=None, weight_add=0, weight_mul=1, from_logits=False):
    '''
    calculates weighted loss
    '''
    if tf.shape(label)[0] == 0:
        return tf.zeros([0], dtype=pred.dtype)

    if weight is None:
        positive_rate = tf_get_positive_rate(label)
        weight = 1 / positive_rate if positive_rate > 0.0 else 1.0

    weight = weight_mul * weight + weight_add
    with tf.control_dependencies([tf.debugging.assert_greater_equal(weight, 0.0, name='assert_on_weight')]):
        weight_mask = label * weight + tf.ones_like(label)

    label = tf.expand_dims(label, -1)
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE, from_logits=from_logits)
    loss = bce(label, pred, sample_weight=weight_mask)
    loss = tf.reduce_mean(loss, [1, 2])
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

    def call(self, y_true, y_pred):
        y_pred_logits = y_pred._keras_logits
        loss = tf_weighted_crossentropy(y_true, y_pred_logits, from_logits=True, **self.config)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config


def tf_get_positive_rate(label):
    max_value = tf.reduce_max(label)
    min_value = tf.reduce_min(label)

    upbound = tf.debugging.assert_less_equal(max_value, 1.0, name='assert_on_max')
    lowbound = tf.debugging.assert_greater_equal(min_value, 0.0, name='assert_on_min')

    with tf.control_dependencies([upbound, lowbound]):
        positive_rate = tf.reduce_sum(label) / tf.cast(tf.reduce_prod(tf.shape(label)), tf.float32)

    assert_range = [
        tf.debugging.assert_greater_equal(positive_rate, 0.0, name='assert_on_range_lower'),
        tf.debugging.assert_less_equal(positive_rate, 1.0, name='assert_on_range_higher'),
    ]
    with tf.control_dependencies(assert_range):
        return positive_rate


tf.keras.utils.get_custom_objects().update(weighted_crossentropy=tf_weighted_crossentropy)
tf.keras.utils.get_custom_objects().update(WeightedCrossentropy=TFWeightedCrossentropy)
