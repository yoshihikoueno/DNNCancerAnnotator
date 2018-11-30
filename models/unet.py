import numpy as np
import tensorflow as tf

from utils import layer_utils as lu
from utils import image_utils


class UNet(object):
  def __init__(self, pipeline_config, is_training):
    assert(pipeline_config.model.input_image_channels == 1)
    self.input_image_dims = (pipeline_config.model.input_image_size_x,
                             pipeline_config.model.input_image_size_y,
                             pipeline_config.model.input_image_channels)
    self.weight_decay = pipeline_config.train_config.weight_decay
    self.use_batch_norm = pipeline_config.model.use_batch_norm
    self.is_training = is_training
    self.config = pipeline_config
    self.num_classes = 1

  def preprocess(self, inputs):
    if inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    if len(inputs.get_shape()) != 4:
      raise ValueError("Expected tensor of rank 4.")

    assert(self.config.dataset.val_range in (0, 1, 2))

    if self.config.dataset.val_range == 1:
      inputs /= 255
    elif self.config.dataset.val_range == 2:
      inputs = (inputs / 255) * 2 - 1

    return inputs

  def _get_conv_params(self, use_relu):
    res = {'data_format': 'channels_last', 'dilation_rate': 1,
           'use_bias': True,
           'kernel_regularizer': tf.keras.regularizers.l2(self.weight_decay),
           'bias_initializer': 'zeros', 'bias_regularizer': None,
           'activity_regularizer': None}
    if use_relu:
      # Variance Scaling is best for relu activations
      res['activation'] = 'relu'
      res['kernel_initializer'] = tf.keras.initializers.VarianceScaling
    else:
      res['kernel_initializer'] = tf.keras.initializers.glorot_uniform

    return res

  def _get_pooling_params(self):
    return {'data_format': 'channels_last'}

  def _get_batch_norm_params(self):
    if self.use_batch_norm:
      return {'epsilon': 0.0001, 'scale': True, 'fused': True,
              'momentum': 0.997, 'center': True, 'trainable': True}
    else:
      return None

  def _downsample_block(self, inputs, nb_filters):
    conv_params = self._get_conv_params(use_relu=True)
    batch_norm_params = self._get_batch_norm_params()
    net = lu.conv_2d(
      inputs=inputs, filters=nb_filters, kernel_size=3, strides=1,
      padding='valid', conv_params=conv_params,
      batch_norm_params=batch_norm_params, is_training=self.is_training)
    net = lu.conv_2d(
      inputs=net, filters=nb_filters, kernel_size=3, strides=1,
      padding='valid', conv_params=conv_params,
      batch_norm_params=batch_norm_params, is_training=self.is_training)

    pool_params = self._get_pooling_params()

    return net, tf.keras.layers.MaxPool2D(pool_size=2, strides=2,
                                          padding='valid', **pool_params)(net)

  def _upsample_block(self, inputs, downsample_reference, nb_filters):
    conv_transposed_params = self._get_conv_params(use_relu=True)
    # For now disable relu in conv transposed
    conv_transposed_params['activation'] = None
    conv_params = self._get_conv_params(use_relu=True)
    batch_norm_params = self._get_batch_norm_params()

    net = lu.conv_2d_transpose(
      inputs=inputs, filters=nb_filters, kernel_size=2, strides=2,
      padding='valid', conv_transposed_params=conv_transposed_params,
      batch_norm_params=batch_norm_params)

    downsample_size = downsample_reference[0].get_shape().as_list()[0]
    target_size = net[0].get_shape().as_list()[0]
    size_difference = downsample_size - target_size

    crop_topleft_y = int(np.floor(size_difference / float(2)))
    crop_topleft_x = int(np.floor(size_difference / float(2)))

    net = tf.concat([net, tf.image.crop_to_bounding_box(
      downsample_reference, crop_topleft_y, crop_topleft_x, target_size,
      target_size)], axis=-1)

    net = lu.conv_2d(
      inputs=net, filters=nb_filters, kernel_size=3, strides=1,
      padding='valid', conv_params=conv_params,
      batch_norm_params=batch_norm_params, is_training=self.is_training)
    net = lu.conv_2d(
      inputs=net, filters=nb_filters, kernel_size=3, strides=1,
      padding='valid', conv_params=conv_params,
      batch_norm_params=batch_norm_params, is_training=self.is_training)

    return net

  def build_network(self, image_batch):
    image_batch = tf.stack(image_batch)
    if (not (image_batch[0].get_shape() == self.input_image_dims)):
      print("Real size of {} is not requested size of {}".format(
        image_batch[0].get_shape(), self.input_image_dims))
      assert(image_batch[0].get_shape() == self.input_image_dims)

    image_batch = self.preprocess(image_batch)

    print(image_batch)
    ds1, pool1 = self._downsample_block(image_batch, 64)
    print(pool1)
    ds2, pool2 = self._downsample_block(pool1, 128)
    print(pool2)
    ds3, pool3 = self._downsample_block(pool2, 256)
    print(pool3)
    ds4, pool4 = self._downsample_block(pool3, 512)
    print(pool4)
    ds5, _ = self._downsample_block(pool4, 1024)
    print(ds5)
    us1 = self._upsample_block(ds5, ds4, 512)
    print(us1)
    us2 = self._upsample_block(us1, ds3, 256)
    print(us2)
    us3 = self._upsample_block(us2, ds2, 128)
    print(us3)
    us4 = self._upsample_block(us3, ds1, 64)
    print(us4)

    conv_params = self._get_conv_params(use_relu=False)

    # TODO: Should we have batch norm here?
    final = lu.conv_2d(inputs=us4, filters=self.num_classes + 1, kernel_size=1,
                       strides=1, padding='valid',
                       is_training=self.is_training, conv_params=conv_params)

    print(final)

    return final

  def loss(self, network_output, groundtruth_mask):
    # GT mask should have batch and channel dimensions
    assert(len(groundtruth_mask.get_shape().as_list()) == 4)

    assert(groundtruth_mask.get_shape().as_list()[:3]
           == network_output.get_shape().as_list()[:3])
    # Mask should be single channel
    assert(groundtruth_mask.get_shape().as_list()[3] == 1)

    mask_loss = tf.losses.sparse_softmax_cross_entropy(
      labels=tf.cast(tf.reshape(groundtruth_mask, [-1]), tf.int32),
      logits=tf.reshape(network_output, [-1, self.num_classes + 1]))

    tf.losses.add_loss(mask_loss)

    return {'mask_loss': mask_loss}
