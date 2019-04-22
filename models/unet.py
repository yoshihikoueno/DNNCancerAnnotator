import functools

import numpy as np
import tensorflow as tf

from utils import layer_utils as lu
from builders import activation_fn_builder as ab


class UNet(object):
  def __init__(self, weight_decay, conv_padding, filter_sizes, down_activation,
               up_activation):
    assert(len(filter_sizes) > 1)
    assert(conv_padding in ['same', 'valid'])
    self.weight_decay = weight_decay
    self.conv_padding = conv_padding
    self.filter_sizes = filter_sizes
    self.down_activation = ab.build(down_activation)
    self.up_activation = ab.build(up_activation)

  def _downsample_block(self, inputs, nb_filters, is_training, use_batch_norm,
                        bn_momentum, bn_epsilon):
    conv_params = lu.get_conv_params(activation_fn=self.down_activation,
                                     weight_decay=self.weight_decay)
    if use_batch_norm:
      batch_norm_params = lu.get_batch_norm_params(momentum=bn_momentum,
                                                   epsilon=bn_epsilon)
    else:
      batch_norm_params = None

    net = lu.conv_2d(
      inputs=inputs, filters=nb_filters, kernel_size=3, strides=1,
      padding=self.conv_padding, conv_params=conv_params,
      batch_norm_params=batch_norm_params, is_training=is_training)
    net = lu.conv_2d(
      inputs=net, filters=nb_filters, kernel_size=3, strides=1,
      padding=self.conv_padding, conv_params=conv_params,
      batch_norm_params=batch_norm_params, is_training=is_training)

    pool_params = lu.get_pooling_params()

    return net, tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2,
                                        padding=self.conv_padding,
                                        **pool_params)

  def _upsample_block(self, inputs, downsample_reference, nb_filters,
                      is_training, use_batch_norm, bn_momentum, bn_epsilon):
    conv_transposed_params = lu.get_conv_transpose_params(
      activation_fn=self.up_activation, weight_decay=self.weight_decay)
    conv_params = lu.get_conv_params(activation_fn=self.up_activation,
                                     weight_decay=self.weight_decay)
    if use_batch_norm:
      batch_norm_params = lu.get_batch_norm_params(momentum=bn_momentum,
                                                   epsilon=bn_epsilon)
    else:
      batch_norm_params = None

    net = lu.conv_2d_transpose(
      inputs=inputs, filters=nb_filters, kernel_size=2, strides=2,
      padding=self.conv_padding, conv_params=conv_transposed_params,
      is_training=is_training, batch_norm_params=batch_norm_params)

    if self.conv_padding == 'valid':
      downsample_size = downsample_reference[0].get_shape().as_list()[0]
      target_size = net[0].get_shape().as_list()[0]
      size_difference = downsample_size - target_size

      crop_topleft_y = int(np.floor(size_difference / float(2)))
      crop_topleft_x = int(np.floor(size_difference / float(2)))

      downsample_reference = tf.image.crop_to_bounding_box(
        downsample_reference, crop_topleft_y, crop_topleft_x, target_size,
        target_size)

    net = tf.concat([net, downsample_reference], axis=-1)

    net = lu.conv_2d(
      inputs=net, filters=nb_filters, kernel_size=3, strides=1,
      padding=self.conv_padding, conv_params=conv_params,
      batch_norm_params=batch_norm_params, is_training=is_training)
    net = lu.conv_2d(
      inputs=net, filters=nb_filters, kernel_size=3, strides=1,
      padding=self.conv_padding, conv_params=conv_params,
      batch_norm_params=batch_norm_params, is_training=is_training)

    return net

  def build_network(self, image_batch, is_training, num_classes,
                    use_batch_norm, bn_momentum, bn_epsilon):
    print(image_batch)

    ds_fn = functools.partial(
      self._downsample_block, is_training=is_training,
      use_batch_norm=use_batch_norm, bn_momentum=bn_momentum,
      bn_epsilon=bn_epsilon)
    us_fn = functools.partial(
      self._upsample_block, is_training=is_training,
      use_batch_norm=use_batch_norm, bn_momentum=bn_momentum,
      bn_epsilon=bn_epsilon)

    with tf.variable_scope('UNet'):
      ds_references = []
      with tf.variable_scope('DownSampleBlock_1'):
        ds, pool = ds_fn(inputs=image_batch,
                         nb_filters=self.filter_sizes[0])
        print(pool)
        ds_references.append(ds)

      for i, filter_size in enumerate(self.filter_sizes[1:-1]):
        with tf.variable_scope('DownSampleBlock_{}'.format(i + 2)):
          ds, pool = ds_fn(inputs=pool, nb_filters=filter_size)
          print(pool)
          ds_references.append(ds)

      with tf.variable_scope('FinalEncodingBlock'):
        us, _ = ds_fn(inputs=pool, nb_filters=self.filter_sizes[-1])
        print(us)

      for i, filter_size in reversed(list(
          enumerate(self.filter_sizes[:-1]))):
        with tf.variable_scope('UpSampleBlock_{}'.format(i + 1)):
          us = us_fn(
            inputs=us, downsample_reference=ds_references[i],
            nb_filters=filter_size)
          print(us)

      conv_params = lu.get_conv_params(activation_fn=None,
                                       weight_decay=self.weight_decay)

      final = lu.conv_2d(inputs=us, filters=num_classes, kernel_size=1,
                         strides=1, padding=self.conv_padding,
                         is_training=is_training, conv_params=conv_params,
                         batch_norm_params=None, name='OutputLayer')
      print(final)

      return final
