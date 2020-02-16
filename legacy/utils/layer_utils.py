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


def conv(inputs, filters, kernel_size, strides, padding, conv_params,
         norm_fn, name=None, norm_first=False,
         is_3d=False, locally_connected=False):
  conv_params = conv_params.copy()

  if norm_first and norm_fn is not None:
    activation = conv_params['activation']
    conv_params['activation'] = None

  if is_3d:
    assert(not locally_connected)
    res = tf.layers.conv3d(
      inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=padding, name=name, **conv_params)
  else:
    if locally_connected:
      del conv_params['dilation_rate']
      res = tf.keras.layers.LocallyConnected2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, implementation=2, **conv_params)(inputs)
    else:
      res = tf.layers.conv2d(
        inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, name=name, **conv_params)

  if norm_fn is not None:
    res = norm_fn(res)

  if norm_first and norm_fn is not None:
    res = activation(res)

  return res


def conv_t(inputs, filters, kernel_size, strides, padding,
           conv_params, norm_fn,
           name=None, norm_first=False, use_dropout=False,
           dropout_rate=0.0, is_3d=False):
  conv_params = conv_params.copy()

  if norm_first and norm_fn is not None:
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

  if norm_fn is not None:
    res = norm_fn(res)

  if norm_first and norm_fn is not None:
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
