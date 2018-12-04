import functools

import tensorflow as tf


def apply(data_augmentation_options, images, gt_masks=None):
  if len(images.get_shape()) != 4:
    raise ValueError("Invalid image dimensions!")
  if gt_masks is not None and len(gt_masks.get_shape()) != 4:
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
                          max_hue=step.random_hue.max_delta), elems=images,
        parallel_iterations=20)
    elif augmentation_type == 'random_saturation':
      images = tf.map_fn(
        functools.partial(tf.image.random_saturation,
                          lower=step.random_saturation.lower,
                          upper=step.random_saturation.upper), elems=images,
        parallel_iterations=20)
    elif augmentation_type == 'random_brightness':
      images = tf.map_fn(
        functools.partial(tf.image.random_brightness,
                          max_hue=step.random_brightness.max_delta),
        elems=images, parallel_iterations=20)

    elif augmentation_type == 'random_warp':
      images, gt_masks = _random_warp(
        images, masks=gt_masks)

    else:
      raise ValueError("Unknown data augmentation type! {}".format(
        augmentation_type))

  return images, gt_masks


def _random_warp(images, masks):
  if masks is not None:
    equal_shape_assert = tf.Assert(tf.equal(images.get_shape()[:3],
                                            masks.get_shape()[:3]))
  else:
    equal_shape_assert = tf.Assert(True)

  with tf.control_dependencies([equal_shape_assert]):
    warp_pts = tf.div(tf.reduce_min(images.get_shape()[1:3]), 2)
    src_control_pts = tf.stack([
      tf.random_uniform([1, warp_pts], minval=0, maxval=images.get_shape()[1]),
      tf.random_uniform([1, warp_pts], minval=0, maxval=images.get_shape()[2])],
                               axis=2)
    target_control_pts = tf.add(
      src_control_pts, tf.random_uniform([1, warp_pts, 2], minval=-5,
                                         maxval=5))

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