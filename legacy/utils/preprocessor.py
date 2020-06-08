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
                raise ValueError(
                    "Expected tensor of rank 5. Got {}".format(inputs))

            assert(not scale_input and "Not yet implemented.")
            assert(val_range == 0)

            features[standard_fields.InputDataFields.image_preprocessed] = inputs

            return features
        else:
            if len(inputs.get_shape()) != 4:
                raise ValueError(
                    "Expected tensor of rank 4. Got {}".format(inputs))

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
                            num_parallel_iterations, is_3d,
                            only_augment_positive):
    if not data_augmentation_options:
        return features

    original_images = features[standard_fields.InputDataFields.image_decoded]
    original_gt_masks = features[standard_fields.InputDataFields.annotation_mask]

    images = original_images
    gt_masks = original_gt_masks

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
                images,
                masks=gt_masks,
                is_3d=is_3d,
                amplitude_ratio=0.05,
                pts_ratio=0.5,
            )

        else:
            raise ValueError(f'Unknown data augmentation type: {augmentation_type}')

    if only_augment_positive:
        if is_3d:
            has_mask = tf.reduce_any(tf.cast(original_gt_masks, tf.bool),
                                     axis=[1, 2, 3, 4])
        else:
            has_mask = tf.reduce_any(tf.cast(original_gt_masks, tf.bool),
                                     axis=[1, 2, 3])

        images = tf.where(has_mask, images, original_images)
        gt_masks = tf.where(has_mask, gt_masks, original_gt_masks)

    features[standard_fields.InputDataFields.image_decoded] = images
    features[standard_fields.InputDataFields.annotation_mask] = gt_masks

    return features


def _random_warp(images, masks, is_3d, amplitude_ratio=0.01, pts_ratio=0.5):
    '''
    apply random warp to the given batch of images.

    This function will take N points randomly on a image,
    and move each of them randomly and independently.

    The number of points to move or "warp" N is determined
    with respect to the size of the image.

    N = pts_ratio * image_size

    note that "image_size" refers to the maximum size of the images.

    image_size = max(image.shape)

    The movement of each points are sampled from the uniform distribution,
    and it's bounds are also determined with respect to the image size.

    Args:
        images: a **batch** of images
        masks: [optional] a batch of masks, can be None
        is_3d: whether the images should be treated as 3D images
        amplitude_ratio: the ratio of maximum warping distance to the image size
        pts_ratio: the ratio of warping points to the image size

    Returns:
        warped images and masks
    '''
    if is_3d:
        dims = 3
    else:
        dims = 2

    max_size = tf.cast(tf.reduce_max(tf.shape(images)), tf.float32)
    num_warp_pts = tf.cast(max_size * pts_ratio, tf.int32)
    amplitude = max_size * amplitude_ratio
    batch_size = images.get_shape().as_list()[0]

    assertions = []
    if masks is not None:
        equal_shape_assert = tf.Assert(
            tf.reduce_all(tf.equal(images.get_shape()[
                          :dims + 1], masks.get_shape()[:dims + 1])),
            data=[images.get_shape(), masks.get_shape()],
        )
        assertions.append(equal_shape_assert)

    with tf.control_dependencies(assertions):
        src_control_pts = tf.stack(
            [
                tf.random_uniform(
                    [batch_size, num_warp_pts],
                    minval=0,
                    maxval=images.get_shape().as_list()[i + 1],
                ) for i in range(dims)
            ], axis=2)

        target_control_pts = tf.random_uniform(
            [batch_size, num_warp_pts, dims],
            minval=-amplitude,
            maxval=amplitude,
        )

        target_control_pts = src_control_pts + target_control_pts
        images, _ = sparse_image_warp.sparse_image_warp(
            images,
            source_control_point_locations=src_control_pts,
            dest_control_point_locations=target_control_pts,
            num_boundary_points=2
        )

        if masks is not None:
            masks, _ = sparse_image_warp.sparse_image_warp(
                masks,
                source_control_point_locations=src_control_pts,
                dest_control_point_locations=target_control_pts,
                num_boundary_points=2,
            )
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