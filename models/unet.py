import logging
import functools

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import standard_fields
from builders import optimizer_builder


def central_crop(img, desired_size):
  assert(len(img.get_shape().as_list()) == 4)
  img_size = img.get_shape().as_list()[1]

  assert(img_size > desired_size)

  offset = int((img_size - desired_size) / 2)

  cropped_img = tf.image.crop_to_bounding_box(img, offset, offset,
                                              desired_size, desired_size)

  return cropped_img


def crop_mask(mask, desired_size):
  return tf.clip_by_value(tf.squeeze(central_crop(
    tf.expand_dims(mask, 3), desired_size), 3), 0, 1)


class UNet(object):
  def __init__(self, pipeline_config, is_training):
    assert(pipeline_config.model.input_image_channels == 1)
    self.input_image_dims = (pipeline_config.model.input_image_size_x,
                             pipeline_config.model.input_image_size_y,
                             pipeline_config.model.input_image_channels)
    self.weight_decay = 0.0
    self.use_batch_norm = False
    self.is_training = is_training
    self.config = pipeline_config

  def _arg_scope(self):
    batch_norm_params = {
      'decay': 0.997,
      'epsilon': 0.0001,
      'scale': True,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'fused': None
    }
    with slim.arg_scope(
        [slim.conv2d], weights_regularizer=slim.l2_regularizer(
          self.weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm if self.use_batch_norm else None,
        normalizer_params=batch_norm_params):
      with slim.arg_scope(
          [slim.conv2d_transpose],
          weights_regularizer=slim.l2_regularizer(self.weight_decay),
          weights_initializer=slim.variance_scaling_initializer(),
          activation_fn=None):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as scope:
          return scope

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

  def _downsample_block(self, inputs, nb_filters):
    net = slim.conv2d(inputs, nb_filters, 3, padding='VALID', stride=1)
    net = slim.conv2d(net, nb_filters, 3, padding='VALID', stride=1)

    return net, slim.max_pool2d(net, 2, stride=2)

  def _upsample_block(self, inputs, downsample_reference, nb_filters):
    net = slim.conv2d_transpose(inputs, nb_filters, 2, stride=2)

    downsample_size = downsample_reference[0].get_shape().as_list()[0]
    target_size = net[0].get_shape().as_list()[0]
    size_difference = downsample_size - target_size

    crop_topleft_y = int(np.floor(size_difference / float(2)))
    crop_topleft_x = int(np.floor(size_difference / float(2)))

    net = tf.concat([net, tf.image.crop_to_bounding_box(
      downsample_reference, crop_topleft_y, crop_topleft_x, target_size,
      target_size)], axis=-1)

    net = slim.conv2d(net, nb_filters, 3, padding='VALID', stride=1)
    net = slim.conv2d(net, nb_filters, 3, padding='VALID', stride=1)

    return net

  def build_network(self, image_batch):
    image_batch = tf.stack(image_batch)
    if (not (image_batch[0].get_shape() == self.input_image_dims)):
      print("Real size of {} is not requested size of {}".format(
        image_batch[0].get_shape(), self.input_image_dims))
      assert(image_batch[0].get_shape() == self.input_image_dims)

    image_batch = self.preprocess(image_batch)

    with tf.variable_scope("UNet", values=[image_batch]):
      with slim.arg_scope(self._arg_scope()):
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

        final = slim.conv2d(us4, 2, 1, padding='VALID', stride=1,
                            activation_fn=None)

        print(final)

        return final

  def loss(self, network_output, groundtruth_mask):
    # Only allow batch size of 1 for now,
    # otherwise we need to check loss calc again
    assert(self.config.train_config.batch_size == 1)

    cropped_groundtruth_mask = crop_mask(
      groundtruth_mask, network_output.get_shape().as_list()[1])

    assert(cropped_groundtruth_mask.get_shape().as_list()[:3]
           == network_output.get_shape().as_list()[:3])

    mask_loss = tf.losses.sparse_softmax_cross_entropy(
      tf.cast(tf.reshape(cropped_groundtruth_mask, [-1]), tf.int32),
      tf.reshape(network_output, [-1, 2]))

    return {'mask_loss': mask_loss}


def estimator_fn(features, pipeline_config, mode):
  if mode == tf.estimator.ModeKeys.TRAIN:
    return train_fn(features, pipeline_config)
  elif mode == tf.estimator.ModeKeys.EVAL:
    return eval_fn(features, pipeline_config)
  else:
    assert(False)


def train_fn(tensor_dict,
             pipeline_config):
  optimizer, optimizer_summary_vars = optimizer_builder.build(
      pipeline_config.train_config.optimizer)

  regularization_losses = (
    None if pipeline_config.train_config.add_regularization_loss else [])

  net = UNet(pipeline_config, is_training=True)

  images = tensor_dict[standard_fields.InputDataFields.image_decoded]
  annotation_mask = tensor_dict[
    standard_fields.InputDataFields.annotation_mask]

  network_output = net.build_network(images)

  losses_dict = net.loss(network_output, annotation_mask)

  total_loss = 0
  for loss_tensor in losses_dict.values():
    tf.losses.add_loss(loss_tensor)
    total_loss += loss_tensor

  grads_and_vars = optimizer.compute_gradients(total_loss)

  # Optionally clip gradients
  if pipeline_config.train_config.gradient_clipping_by_norm > 0:
    grads_and_vars = slim.learning.clip_gradient_norms(
      grads_and_vars, pipeline_config.train_config.gradient_clipping_by_norm)

  train_op = optimizer.apply_gradients(
    grads_and_vars, global_step=tf.train.get_global_step())

  logging.info("Total number of trainable parameters: {}".format(np.sum([
    np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

  return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN,
                                    loss=total_loss, train_op=train_op)


def eval_fn(queue, pipeline_config):
  input_dict = queue.dequeue()
  images = tf.to_float(tf.expand_dims(
    input_dict[fields.InputDataFields.image], 0))

  net = UNet(pipeline_config, is_training=False)

  network_output = net.build_network(images)

  groundtruth = {
    fields.InputDataFields.groundtruth_boxes:
      input_dict[fields.InputDataFields.groundtruth_boxes],
    fields.InputDataFields.groundtruth_classes:
      input_dict[fields.InputDataFields.groundtruth_classes],
    fields.InputDataFields.groundtruth_difficult:
      input_dict[fields.InputDataFields.groundtruth_difficult],
    fields.InputDataFields.groundtruth_instance_masks:
      input_dict[fields.InputDataFields.groundtruth_instance_masks]
  }

  return {'network_input': images,
          'network_output': network_output,
          'key': input_dict[fields.InputDataFields.source_id], 'gt_masks':
          tf.expand_dims(groundtruth[fields.InputDataFields.groundtruth_instance_masks], axis=0)}, {}


def eval_batch_processor_fn(eval_config, categories, result_folder, tensor_dict,
                            sess, batch_index, counters, losses_dict=None):
  if (losses_dict is None):
    losses_dict = {}
  global_step = tf.train.global_step(sess, tf.train.get_global_step())

  network_output = tensor_dict['network_output']

  background, foreground = tf.split(tf.sigmoid(network_output), 2, axis=3)
  img_size = background.get_shape().as_list()[1]

  gt_mask = tf.expand_dims(combine_and_crop_masks(tensor_dict['gt_masks'], img_size), axis=3)

  background = tf.tile(background, [1, 1, 1, 3]) * 255
  foreground = tf.tile(foreground, [1, 1, 1, 3]) * 255
  gt_mask = tf.concat([gt_mask, tf.zeros_like(gt_mask), tf.zeros_like(gt_mask)], axis = 3) * 255
  original_image = central_crop(tf.tile(tensor_dict['network_input'], [1, 1, 1, 3]), img_size)

  masked_img = tf.clip_by_value(original_image + gt_mask, 0, 255)

  side_by_side = tf.concat([background, foreground, masked_img], axis=2)

  result_dict, result_losses_dict, side_by_side_result = sess.run(
    [tensor_dict, losses_dict, side_by_side])

  image = np.squeeze(side_by_side_result, axis=0)

  summary = tf.Summary(value=[
          tf.Summary.Value(
            tag=result_dict['key'],
            image=tf.Summary.Image(
              encoded_image_string=vis_utils.encode_image_array_as_png_str(
                image)))
          ])

  summary_writer = tf.summary.FileWriterCache.get(result_folder)
  summary_writer.add_summary(summary, global_step)

  logging.info('Detection visualizations written to summary with tag %s.',
                       result_dict['key'])
  counters['success'] += 1

  return result_dict, result_losses_dict
