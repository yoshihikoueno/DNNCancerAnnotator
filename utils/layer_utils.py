import tensorflow as tf


def get_pooling_params():
  return {'data_format': 'channels_last'}


def get_conv_params(activation_fn, weight_decay):
  res = {'data_format': 'channels_last', 'dilation_rate': 1,
         'use_bias': True,
         'kernel_regularizer': (tf.contrib.layers.l2_regularizer(weight_decay)
                                if weight_decay > 0 else None),
         'bias_initializer': tf.zeros_initializer(), 'bias_regularizer': None,
         'activity_regularizer': None, 'activation': activation_fn}

  res['kernel_initializer'] = tf.keras.initializers.VarianceScaling(
      scale=2.0)

  return res


def get_conv_transpose_params(activation_fn, weight_decay):
  res = {'data_format': 'channels_last',
         'use_bias': True,
         'kernel_regularizer': (tf.contrib.layers.l2_regularizer(weight_decay)
                                if weight_decay > 0 else None),
         'bias_initializer': tf.zeros_initializer(), 'bias_regularizer': None,
         'activity_regularizer': None, 'activation': activation_fn,
         'kernel_initializer': tf.keras.initializers.VarianceScaling(
           scale=2.0)}

  return res


def get_batch_norm_params(momentum, epsilon):
  return {'epsilon': epsilon, 'scale': True, 'fused': True,
          'momentum': momentum, 'center': True, 'trainable': True}


def conv(inputs, filters, kernel_size, strides, padding, conv_params,
         batch_norm_params, is_training, name=None, bn_first=False,
         is_3d=False):
  conv_params = conv_params.copy()

  if bn_first:
    activation = conv_params['activation']
    conv_params['activation'] = None

  if is_3d:
    res = tf.layers.conv3d(
      inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=padding, name=name, **conv_params)
  else:
    res = tf.layers.conv2d(
      inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=padding, name=name, **conv_params)

  if batch_norm_params is not None:
    axis = -1 if conv_params['data_format'] == 'channels_last' else 1
    res = tf.layers.batch_normalization(
      inputs=res, axis=axis, name=name + 'BN' if name is not None else None,
      **batch_norm_params, training=is_training)

  if bn_first:
    res = activation(res)

  return res


def conv_t(inputs, filters, kernel_size, strides, padding,
           conv_params, batch_norm_params,
           is_training, name=None, bn_first=False, use_dropout=False,
           dropout_rate=0.0, is_3d=False):
  conv_params = conv_params.copy()

  if bn_first:
    activation = conv_params['activation']
    conv_params['activation'] = None

  if is_3d:
    res = tf.layers.conv3d_transpose(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=padding, name=name, **conv_params)
  else:
    res = tf.layers.conv2d_transpose(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=padding, name=name, **conv_params)

  if batch_norm_params is not None:
    axis = -1 if conv_params['data_format'] == 'channels_last' else 1
    res = tf.layers.batch_normalization(
      inputs=res, axis=axis, name=name + 'BN' if name is not None else None,
      training=is_training, **batch_norm_params)

  if bn_first:
    res = activation(res)

  if use_dropout:
    res = tf.nn.dropout(res, rate=dropout_rate, name=(
      name + 'Dropout' if name is not None else None))

  return res


def pool(inputs, pool_size, strides, padding, pool_params, is_3d=False):
  if is_3d:
    return tf.layers.max_pooling3d(
      inputs=inputs, pool_size=pool_size, strides=strides,
      padding=padding, **pool_params)
  else:
    return tf.layers.max_pooling2d(
      inputs=inputs, pool_size=pool_size, strides=strides,
      padding=padding, **pool_params)
