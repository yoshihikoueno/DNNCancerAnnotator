import functools

import tensorflow as tf

from utils import layer_utils as lu


class GANDiscriminator(object):
  def __init__(self, conv_bn_first):
    self.conv_bn_first = conv_bn_first

  def build_network(self, image_batch, is_training, num_classes,
                    use_batch_norm, bn_momentum, bn_epsilon):
    image_shape = image_batch.get_shape().as_list()

    # We need to move across the image and classify all patches, then
    # average the result
    #tf.image.extract_image_patches(image_batch, ksizes=[1, 70, 70,
    #                                                    image_shape[3]])

    assert(image_shape[3] == 2)

    with tf.variable_scope('GANDiscriminator'):
      # Default is 3 layers
      # Filter sizes are 64, 128, 256, 512
      # Conv layer applied at the end to map to 1 dim output, followed by sigmoid
      # Batch norm is not applied to the layer with 64 filters
      # Using leaky relu with slope 0.2
      conv_params = lu.get_conv_params(functools.partial(
        tf.nn.leaky_relu, alpha=0.2), weight_decay=0.0)
      conv_params_no_af = lu.get_conv_params(None, 0.0)
      if use_batch_norm:
        batch_norm_params = lu.get_batch_norm_params(
          momentum=bn_momentum, epsilon=bn_epsilon)
      else:
        batch_norm_params = None

      print(image_batch)

      c1 = lu.conv(image_batch, filters=64, kernel_size=4, strides=2,
                   padding='same', conv_params=conv_params,
                   batch_norm_params=None, is_training=is_training,
                   name='conv1')
      print(c1)
      c2 = lu.conv(c1, filters=128, kernel_size=4, strides=2,
                      padding='same', conv_params=conv_params,
                      batch_norm_params=batch_norm_params,
                      is_training=is_training, name='conv2')
      print(c2)
      c3 = lu.conv(c2, filters=256, kernel_size=4, strides=2,
                      padding='same', conv_params=conv_params,
                      batch_norm_params=batch_norm_params,
                      is_training=is_training, name='conv3')
      print(c3)
      c4 = lu.conv(c3, filters=512, kernel_size=4, strides=2,
                      padding='same', conv_params=conv_params,
                      batch_norm_params=batch_norm_params,
                      is_training=is_training, name='conv4')
      print(c4)
      c5 = lu.conv(c4, filters=512, kernel_size=4, strides=2,
                   padding='same', conv_params=conv_params,
                   batch_norm_params=batch_norm_params,
                   is_training=is_training, name='conv5')
      print(c5)
      c6 = lu.conv(c5, filters=512, kernel_size=4, strides=2,
                   padding='same', conv_params=conv_params,
                   batch_norm_params=batch_norm_params,
                   is_training=is_training, name='conv6')
      print(c6)
      output = lu.conv(c6, filters=1, kernel_size=4, strides=1,
                       padding='same', conv_params=conv_params_no_af,
                       batch_norm_params=None,
                       is_training=is_training, name='output')

      output = tf.reduce_mean(output, axis=[1, 2, 3])

      print(output)

      return tf.nn.sigmoid(output)
