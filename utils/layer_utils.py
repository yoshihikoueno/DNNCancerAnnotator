import tensorflow as tf


def get_pooling_params():
  return {'data_format': 'channels_last'}


def get_conv_params(use_relu, weight_decay):
  res = {'data_format': 'channels_last', 'dilation_rate': 1,
         'use_bias': True,
         'kernel_regularizer': tf.keras.regularizers.l2(weight_decay),
         'bias_initializer': 'zeros', 'bias_regularizer': None,
         'activity_regularizer': None}
  if use_relu:
    # Variance Scaling is best for relu activations
    res['activation'] = 'relu'
    res['kernel_initializer'] = tf.keras.initializers.VarianceScaling(
      scale=2.0)
  else:
    #res['kernel_initializer'] = tf.keras.initializers.glorot_uniform()
    res['activation'] = None
    res['kernel_initializer'] = tf.keras.initializers.VarianceScaling(
      scale=2.0)

  return res


def get_conv_transpose_params(weight_decay):
  res = {'data_format': 'channels_last', 'dilation_rate': 1,
         'use_bias': True,
         'kernel_regularizer': tf.keras.regularizers.l2(weight_decay),
         'bias_initializer': 'zeros', 'bias_regularizer': None,
         'activity_regularizer': None, 'activation': None,
         'kernel_initializer': tf.keras.initializers.VarianceScaling(
           scale=2.0)}

  return res


def get_batch_norm_params(momentum, epsilon):
  return {'epsilon': epsilon, 'scale': True, 'fused': True,
          'momentum': momentum, 'center': True, 'trainable': True}


def conv_2d(inputs, filters, kernel_size, strides, padding, conv_params,
            batch_norm_params, is_training, name=None):
  res = tf.keras.layers.Conv2D(
    filters=filters, kernel_size=kernel_size, strides=strides,
    padding=padding, name=name, **conv_params)(inputs)

  if batch_norm_params:
    axis = -1 if conv_params['data_format'] == 'channels_last' else 1
    res = tf.layers.batch_normalization(
      inputs=res, axis=axis, name=name + 'BN' if name is not None else None,
      **batch_norm_params, training=is_training)

  return res


def conv_2d_transpose(inputs, filters, kernel_size, strides, padding,
                      conv_params, batch_norm_params,
                      is_training, name=None):
  res = tf.keras.layers.Conv2DTranspose(
    filters=filters, kernel_size=kernel_size, strides=strides,
    padding=padding, name=name, **conv_params)(inputs)

  if batch_norm_params:
    axis = -1 if conv_params['data_format'] == 'channels_last' else 1
    res = tf.layers.batch_normalization(
      inputs=res, axis=axis, name=name + 'BN' if name is not None else None,
      training=is_training, **batch_norm_params)

  return res
