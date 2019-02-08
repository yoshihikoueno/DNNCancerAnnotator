import functools

import tensorflow as tf


def build(model_config):
  if not hasattr(model_config, 'activation'):
    return tf.nn.relu

  activation_name = model_config.activation.WhichOneof('activation_fn')

  if activation_name == 'relu' or activation_name is None:
    return tf.nn.relu
  elif activation_name == 'leaky_relu':

    return functools.partial(tf.nn.leaky_relu,
                             alpha=model_config.activation.leaky_relu.alpha)
  else:
    raise ValueError("Invalid activation function {}".format(activation_name))
