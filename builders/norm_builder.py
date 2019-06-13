import functools

import tensorflow as tf


def build(config, is_training):
  norm_type = config.WhichOneof('norm')

  if norm_type == 'batch_norm':
    epsilon = config.batch_norm.epsilon
    momentum = config.batch_norm.momentum
    return functools.partial(tf.layers.batch_normalization, epsilon=epsilon,
                             momentum=momentum, scale=True, fused=True,
                             center=True, trainable=True, axis=-1,
                             is_training=is_training)
  elif norm_type == 'instance_norm':
    return functools.partial(tf.contrib.layers.instance_norm, scale=True,
                             center=True, trainable=True)
  else:
    raise ValueError("Invalid normalization {}".format(norm_type))
