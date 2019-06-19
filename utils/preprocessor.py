import functools

import tensorflow as tf

from utils import util_ops
from utils import standard_fields
from utils import sparse_image_warp


def preprocess(features, val_range, scale_input, model_objective,
               tfrecords_type):
  if model_objective == 'segmentation':
    inputs = features[standard_fields.InputDataFields.image_decoded]
    if inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')

    if tfrecords_type == 'input_3d':
      if len(inputs.get_shape()) != 5:
        raise ValueError("Expected tensor of rank 5. Got {}".format(inputs))

      assert(not scale_input and "Not yet implemented.")
      assert(val_range == 0)

      features[standard_fields.InputDataFields.image_preprocessed] = inputs

      return features
    else:
      if len(inputs.get_shape()) != 4:
        raise ValueError("Expected tensor of rank 4. Got {}".format(inputs))

      assert(val_range in (0, 1, 2))

      if scale_input:
        inputs = tf.map_fn(tf.image.per_image_standardization, elems=inputs,
                           parallel_iterations=util_ops.get_cpu_count())
      else:
        # We don't care about the val_range if we scale our input, since it would
        # be the same anyway
        if val_range == 1:
          inputs /= 255
        elif val_range == 2:
          inputs = (inputs / 255) * 2 - 1

      features[standard_fields.InputDataFields.image_preprocessed] = inputs

      return features
  elif model_objective == 'interpolation':
    # Not yet implemented
    assert(not scale_input and val_range == 0)

    features[standard_fields.InputDataFields.image_preprocessed] = features[
      standard_fields.InputDataFields.image_decoded]

    return features


def apply_data_augmentation(features, data_augmentation_options,
                            num_parallel_iterations, is_3d):
  if not data_augmentation_options:
    return features

  images = features[standard_fields.InputDataFields.image_decoded]
  gt_masks = features[standard_fields.InputDataFields.annotation_mask]

  required_dims = 5 if is_3d else 4
  if len(images.get_shape()) != required_dims:
    raise ValueError("Invalid image dimensions!")
  if gt_masks is not None and len(gt_masks.get_shape()) != required_dims:
    raise ValueError("Invalid mask dimensions!")

  applied_augmentations = set()

  for step in data_augmentation_options:
    augmentation_type = step.WhichOneof('preprocessing_step')
    if augmentation_type is None:
      continue
    if augmentation_type in applied_augmentations:
      raise ValueError(
        "Duplicate augmentation type! {}".format(augmentation_type))

    applied_augmentations.add(augmentation_type)

    if augmentation_type == 'random_horizontal_flip':
      images, gt_masks = _random_horizontal_flip(
        images, masks=gt_masks)

    elif augmentation_type == 'random_vertical_flip':
      images, gt_masks = _random_vertical_flip(
        images, masks=gt_masks)

    elif augmentation_type == 'random_contrast':
      images = tf.map_fn(
        functools.partial(tf.image.random_contrast,
                          lower=step.random_contrast.lower,
                          upper=step.random_contrast.upper), elems=images,
        parallel_iterations=20)

    elif augmentation_type == 'random_hue':
      images = tf.map_fn(
        functools.partial(tf.image.random_hue,
                          max_delta=step.random_hue.max_delta), elems=images,
        parallel_iterations=num_parallel_iterations)

    elif augmentation_type == 'random_saturation':
      images = tf.map_fn(
        functools.partial(tf.image.random_saturation,
                          lower=step.random_saturation.lower,
                          upper=step.random_saturation.upper), elems=images,
        parallel_iterations=num_parallel_iterations)

    elif augmentation_type == 'random_brightness':
      images = tf.map_fn(
        functools.partial(tf.image.random_brightness,
                          max_delta=step.random_brightness.max_delta),
        elems=images, parallel_iterations=num_parallel_iterations)

    elif augmentation_type == 'random_warp':
      images, gt_masks = _random_warp(
        images, masks=gt_masks, is_3d=is_3d)

    else:
      raise ValueError("Unknown data augmentation type! {}".format(
        augmentation_type))

  features[standard_fields.InputDataFields.image_decoded] = images
  features[standard_fields.InputDataFields.annotation_mask] = gt_masks

  return features


def _random_warp(images, masks, is_3d):
  if is_3d:
    dims = 3
  else:
    dims = 2

  if masks is not None:
    equal_shape_assert = tf.Assert(tf.reduce_all(tf.equal(
      images.get_shape()[:dims + 1], masks.get_shape()[:dims + 1])), data=[
        images.get_shape(), masks.get_shape()])
  else:
    equal_shape_assert = tf.Assert(True, data=[])

  with tf.control_dependencies([equal_shape_assert]):
    num_warp_pts = tf.cast(
      tf.divide(tf.reduce_max(tf.shape(images)), 2), tf.int32)
    if is_3d:
      src_control_pts = tf.stack([
        tf.random_uniform([images.get_shape()[0], num_warp_pts], minval=0,
                          maxval=images.get_shape().as_list()[1]),
        tf.random_uniform([images.get_shape()[0], num_warp_pts], minval=0,
                          maxval=images.get_shape().as_list()[1]),
        tf.random_uniform([images.get_shape()[0], num_warp_pts], minval=0,
                          maxval=images.get_shape().as_list()[2])], axis=2)

      target_control_pts = tf.stack([
        tf.random_uniform([images.get_shape()[0], num_warp_pts],
                          minval=-2, maxval=2),
        tf.random_uniform([images.get_shape()[0], num_warp_pts],
                          minval=-5, maxval=5),
        tf.random_uniform([images.get_shape()[0], num_warp_pts],
                          minval=-5, maxval=5)
      ], axis=2)
      target_control_pts = tf.add(src_control_pts, target_control_pts)

      images, _ = sparse_image_warp.sparse_image_warp(
        images, source_control_point_locations=src_control_pts,
        dest_control_point_locations=target_control_pts,
        num_boundary_points=2)

      if masks is not None:
        masks, _ = sparse_image_warp.sparse_image_warp(
          masks, source_control_point_locations=src_control_pts,
          dest_control_point_locations=target_control_pts,
          num_boundary_points=2)

        # We are only interested in values 0 or 1
        masks = tf.to_float(masks > 0.5)
    else:
      src_control_pts = tf.stack([
        tf.random_uniform([images.get_shape()[0], num_warp_pts], minval=0,
                          maxval=images.get_shape().as_list()[1]),
        tf.random_uniform([images.get_shape()[0], num_warp_pts], minval=0,
                          maxval=images.get_shape().as_list()[2])], axis=2)

      target_control_pts = tf.stack([
        tf.random_uniform([images.get_shape()[0], num_warp_pts],
                          minval=-5, maxval=5),
        tf.random_uniform([images.get_shape()[0], num_warp_pts],
                          minval=-5, maxval=5)], axis=2)
      target_control_pts = tf.add(src_control_pts, target_control_pts)

      images, _ = tf.contrib.image.sparse_image_warp(
        images, source_control_point_locations=src_control_pts,
        dest_control_point_locations=target_control_pts,
        num_boundary_points=2)

      if masks is not None:
        masks, _ = tf.contrib.image.sparse_image_warp(
          masks, source_control_point_locations=src_control_pts,
          dest_control_point_locations=target_control_pts,
          num_boundary_points=2)

        # We are only interested in values 0 or 1
        masks = tf.to_float(masks > 0.5)

  return images, masks


def _random_vertical_flip(images, masks):
  # random variable defining whether to do flip or not
  generator_func = functools.partial(tf.random_uniform, [])
  do_a_flip_random = generator_func()
  do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

  images = tf.cond(do_a_flip_random, lambda: tf.image.flip_up_down(images),
                   lambda: images)
  if masks is not None:
    masks = tf.cond(do_a_flip_random, lambda: tf.image.flip_up_down(masks),
                    lambda: masks)

  return images, masks


def _random_horizontal_flip(images, masks):
  # random variable defining whether to do flip or not
  generator_func = functools.partial(tf.random_uniform, [])
  do_a_flip_random = generator_func()
  do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

  images = tf.cond(do_a_flip_random, lambda: tf.image.flip_left_right(images),
                   lambda: images)
  if masks is not None:
    masks = tf.cond(do_a_flip_random, lambda: tf.image.flip_left_right(masks),
                    lambda: masks)

  return images, masks
