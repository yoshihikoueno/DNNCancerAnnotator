import functools

import tensorflow as tf


def build(config, is_training, is_3d):
    norm_type = config.WhichOneof('norm')

    if norm_type == 'batch_norm':
        epsilon = config.batch_norm.epsilon
        momentum = config.batch_norm.momentum
        return functools.partial(tf.layers.batch_normalization, epsilon=epsilon,
                                 momentum=momentum, scale=True, fused=True,
                                 center=True, trainable=True, axis=-1,
                                 training=is_training)
    elif norm_type == 'instance_norm':
        return functools.partial(tf.contrib.layers.instance_norm, scale=True,
                                 center=True, trainable=True)
    elif norm_type == 'group_norm':
        groups = config.group_norm.groups
        reduction_axes = (-4, -3, -2) if is_3d else (-3, -2)
        return functools.partial(
            tf.contrib.layers.group_norm, groups=groups, channels_axis=-1,
            reduction_axes=reduction_axes, center=True, scale=True, trainable=True)
    elif norm_type == 'layer_norm':
        return functools.partial(
            tf.contrib.layers.layer_norm, center=True, scale=True, trainable=True)
    else:
        raise ValueError("Invalid normalization {}".format(norm_type))
