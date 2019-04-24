import functools

import tensorflow as tf

from utils import layer_utils as lu


class UNetP2P(object):
  def __init__(self, conv_bn_first):
    self.conv_bn_first = conv_bn_first

  def build_network(self, image_batch, is_training, num_classes,
                    use_batch_norm, bn_momentum, bn_epsilon):
    print(image_batch)

    with tf.variable_scope('UNetP2P'):
      conv_params_leaky = lu.get_conv_params(functools.partial(
        tf.nn.leaky_relu, alpha=0.2), weight_decay=0.0)
      conv_transpose_tanh_params = lu.get_conv_transpose_params(
        activation_fn=tf.nn.tanh, weight_decay=0.0)
      conv_transposed_params = lu.get_conv_transpose_params(
        activation_fn=tf.nn.relu, weight_decay=0.0)
      if use_batch_norm:
        batch_norm_params = lu.get_batch_norm_params(
          momentum=bn_momentum, epsilon=bn_epsilon)
      else:
        batch_norm_params = None

      c1 = lu.conv(image_batch, filters=64, kernel_size=4, strides=2,
                      padding='same', conv_params=conv_params_leaky,
                      batch_norm_params=None, is_training=is_training,
                      name='conv1', bn_first=self.conv_bn_first)
      print(c1)
      c2 = lu.conv(c1, filters=128, kernel_size=4, strides=2,
                   padding='same', conv_params=conv_params_leaky,
                   batch_norm_params=batch_norm_params,
                   is_training=is_training, name='conv2',
                   bn_first=self.conv_bn_first)
      print(c2)
      c3 = lu.conv(c2, filters=256, kernel_size=4, strides=2,
                   padding='same', conv_params=conv_params_leaky,
                   batch_norm_params=batch_norm_params,
                   is_training=is_training, name='conv3',
                   bn_first=self.conv_bn_first)
      print(c3)
      c4 = lu.conv(c3, filters=512, kernel_size=4, strides=2,
                   padding='same', conv_params=conv_params_leaky,
                   batch_norm_params=batch_norm_params,
                   is_training=is_training, name='conv4',
                   bn_first=self.conv_bn_first)
      print(c4)
      c5 = lu.conv(c4, filters=512, kernel_size=4, strides=2,
                   padding='same', conv_params=conv_params_leaky,
                   batch_norm_params=batch_norm_params,
                   is_training=is_training, name='conv5',
                   bn_first=self.conv_bn_first)
      print(c5)
      c6 = lu.conv(c5, filters=512, kernel_size=4, strides=2,
                   padding='same', conv_params=conv_params_leaky,
                   batch_norm_params=batch_norm_params,
                   is_training=is_training, name='conv6',
                   bn_first=self.conv_bn_first)
      print(c6)
      c7 = lu.conv(c6, filters=512, kernel_size=4, strides=2,
                   padding='same', conv_params=conv_params_leaky,
                   batch_norm_params=batch_norm_params,
                   is_training=is_training, name='conv7',
                   bn_first=self.conv_bn_first)
      print(c7)
      c8 = lu.conv(c7, filters=512, kernel_size=4, strides=2,
                   padding='same', conv_params=conv_params_leaky,
                   batch_norm_params=batch_norm_params,
                   is_training=is_training, name='conv8',
                   bn_first=self.conv_bn_first)
      print(c8)
      u1 = lu.conv_t(c8, filters=512, kernel_size=4, strides=2, padding='same',
                     conv_params=conv_transposed_params,
                     batch_norm_params=batch_norm_params,
                     is_training=is_training, name='up1',
                     bn_first=self.conv_bn_first, use_dropout=True,
                     dropout_rate=0.5)
      u1 = tf.concat([u1, c7], axis=-1)
      print(u1)
      u2 = lu.conv_t(u1, filters=512, kernel_size=4, strides=2, padding='same',
                     conv_params=conv_transposed_params,
                     batch_norm_params=batch_norm_params,
                     is_training=is_training, name='up2',
                     bn_first=self.conv_bn_first, use_dropout=True,
                     dropout_rate=0.5)
      u2 = tf.concat([u2, c6], axis=-1)
      print(u2)
      u3 = lu.conv_t(u2, filters=512, kernel_size=4, strides=2, padding='same',
                     conv_params=conv_transposed_params,
                     batch_norm_params=batch_norm_params,
                     is_training=is_training, name='up3',
                     bn_first=self.conv_bn_first, use_dropout=True,
                     dropout_rate=0.5)
      u3 = tf.concat([u3, c5], axis=-1)
      print(u3)
      u4 = lu.conv_t(u3, filters=512, kernel_size=4, strides=2, padding='same',
                     conv_params=conv_transposed_params,
                     batch_norm_params=batch_norm_params,
                     is_training=is_training, name='up4',
                     bn_first=self.conv_bn_first)
      u4 = tf.concat([u4, c4], axis=-1)
      print(u4)
      u5 = lu.conv_t(u4, filters=256, kernel_size=4, strides=2, padding='same',
                     conv_params=conv_transposed_params,
                     batch_norm_params=batch_norm_params,
                     is_training=is_training, name='up5',
                     bn_first=self.conv_bn_first)
      u5 = tf.concat([u5, c3], axis=-1)
      print(u5)
      u6 = lu.conv_t(u5, filters=128, kernel_size=4, strides=2, padding='same',
                     conv_params=conv_transposed_params,
                     batch_norm_params=batch_norm_params,
                     is_training=is_training, name='up6',
                     bn_first=self.conv_bn_first)
      u6 = tf.concat([u6, c2], axis=-1)
      print(u6)
      u7 = lu.conv_t(u6, filters=64, kernel_size=4, strides=2, padding='same',
                     conv_params=conv_transposed_params,
                     batch_norm_params=batch_norm_params,
                     is_training=is_training, name='up7',
                     bn_first=self.conv_bn_first)
      u7 = tf.concat([u7, c1], axis=-1)
      print(u7)

      final = lu.conv_t(u7, filters=1, kernel_size=4, strides=2,
                        padding='same',
                        conv_params=conv_transpose_tanh_params,
                        batch_norm_params=batch_norm_params,
                        is_training=is_training, name='final',
                        bn_first=self.conv_bn_first)

      print(final)

      return final
