import numpy as np
import tensorflow as tf

from utils import layer_utils as lu


class UNet(object):
  def __init__(self, weight_decay, use_batch_norm, conv_padding,
               final_filter_size, num_sample_steps):
    assert(num_sample_steps > 0)
    assert(conv_padding in ['same', 'valid'])
    self.weight_decay = weight_decay
    self.use_batch_norm = use_batch_norm
    self.conv_padding = conv_padding
    self.final_filter_size = final_filter_size
    self.num_sample_steps = num_sample_steps

  def _downsample_block(self, inputs, nb_filters, is_training):
    conv_params = lu.get_conv_params(use_relu=True,
                                     weight_decay=self.weight_decay)
    if self.use_batch_norm:
      batch_norm_params = lu.get_batch_norm_params()
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

    return net, tf.keras.layers.MaxPool2D(pool_size=2, strides=2,
                                          padding=self.conv_padding,
                                          **pool_params)(net)

  def _upsample_block(self, inputs, downsample_reference, nb_filters,
                      is_training):
    conv_transposed_params = lu.get_conv_params(use_relu=True,
                                                weight_decay=self.weight_decay)
    # For now disable relu in conv transposed
    conv_transposed_params['activation'] = None
    conv_params = lu.get_conv_params(use_relu=True,
                                     weight_decay=self.weight_decay)
    if self.use_batch_norm:
      batch_norm_params = lu.get_batch_norm_params()
    else:
      batch_norm_params = None

    net = lu.conv_2d_transpose(
      inputs=inputs, filters=nb_filters, kernel_size=2, strides=2,
      padding=self.conv_padding, conv_transposed_params=conv_transposed_params,
      batch_norm_params=batch_norm_params, is_training=is_training)

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

  def build_network(self, image_batch, is_training, num_classes):
    filter_size_per_block = []
    for i in range(self.num_sample_steps):
      filter_size_per_block.append(
        int(self.final_filter_size / (2**(self.num_sample_steps - i))))

    print(image_batch)

    ds_references = []
    ds, pool = self._downsample_block(image_batch, filter_size_per_block[0],
                                      is_training=is_training)
    print(pool)
    ds_references.append(ds)
    for filter_size in filter_size_per_block[1:]:
      ds, pool = self._downsample_block(pool, filter_size,
                                        is_training=is_training)
      print(pool)
      ds_references.append(ds)

    us, _ = self._downsample_block(pool, self.final_filter_size,
                                         is_training=is_training)
    print(us)

    for i, filter_size in reversed(list(
        enumerate(filter_size_per_block))):
      us = self._upsample_block(us, ds_references[i], filter_size,
                                is_training=is_training)
      print(us)

    conv_params = lu.get_conv_params(use_relu=False,
                                     weight_decay=self.weight_decay)

    # TODO: Should we have batch norm here?
    final = lu.conv_2d(inputs=us, filters=num_classes, kernel_size=1,
                       strides=1, padding=self.conv_padding,
                       is_training=is_training, conv_params=conv_params,
                       batch_norm_params=None)
    print(final)

    return final
