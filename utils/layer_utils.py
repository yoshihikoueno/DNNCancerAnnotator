import tensorflow as tf


def conv_2d(inputs, filters, kernel_size, strides, padding, conv_params,
            batch_norm_params=None, is_training=False):
  res = tf.keras.layers.Conv2D(
    filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
    **conv_params)(inputs)

  if batch_norm_params:
    axis = 3 if conv_params['data_format'] == 'channels_last' else 1
    res = tf.keras.layers.BatchNormalization(
      axis=axis, **batch_norm_params)(res, training=is_training)

  return res


def conv_2d_transpose(inputs, filters, kernel_size, strides, padding,
                      conv_transposed_params, batch_norm_params=None,
                      is_training=False):
  res = tf.keras.layers.Conv2DTranspose(
    filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
    **conv_transposed_params)(inputs)

  if batch_norm_params:
    axis = 3 if conv_transposed_params['data_format'] == 'channels_last' else 1
    res = tf.keras.layers.BatchNormalization(
      axis=axis, **batch_norm_params)(res, training=is_training)

  return res
