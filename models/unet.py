import numpy as np
import tensorflow as tf

from utils import layer_utils as lu


class UNet(object):
  def __init__(self, weight_decay, use_batch_norm, conv_padding):
    assert(conv_padding in ['same', 'valid'])
    self.weight_decay = weight_decay
    self.use_batch_norm = use_batch_norm
    self.conv_padding = conv_padding

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
      batch_norm_params=batch_norm_params)

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
    print(image_batch)
    ds1, pool1 = self._downsample_block(image_batch, 64,
                                        is_training=is_training)
    print(pool1)
    ds2, pool2 = self._downsample_block(pool1, 128, is_training=is_training)
    print(pool2)
    ds3, pool3 = self._downsample_block(pool2, 256, is_training=is_training)
    print(pool3)
    ds4, pool4 = self._downsample_block(pool3, 512, is_training=is_training)
    print(pool4)
    ds5, _ = self._downsample_block(pool4, 1024, is_training=is_training)
    print(ds5)
    us1 = self._upsample_block(ds5, ds4, 512, is_training=is_training)
    print(us1)
    us2 = self._upsample_block(us1, ds3, 256, is_training=is_training)
    print(us2)
    us3 = self._upsample_block(us2, ds2, 128, is_training=is_training)
    print(us3)
    us4 = self._upsample_block(us3, ds1, 64, is_training=is_training)
    print(us4)

    conv_params = lu.get_conv_params(use_relu=False,
                                     weight_decay=self.weight_decay)

    # TODO: Should we have batch norm here?
    final = lu.conv_2d(inputs=us4, filters=num_classes, kernel_size=1,
                       strides=1, padding=self.conv_padding,
                       is_training=is_training, conv_params=conv_params)
    print(final)

    return final
