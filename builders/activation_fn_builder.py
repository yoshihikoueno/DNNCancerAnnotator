import functools

import tensorflow as tf


def build(activation_config):
  activation_name = activation_config.WhichOneof('activation_fn')

  if activation_name == 'relu' or activation_name is None:
    return tf.nn.relu
  elif activation_name == 'leaky_relu':

    return functools.partial(tf.nn.leaky_relu,
                             alpha=activation_config.leaky_relu.alpha)
  else:
    raise ValueError("Invalid activation function {}".format(activation_name))
