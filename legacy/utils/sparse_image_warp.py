import numpy as np

import tensorflow as tf
from tensorflow.contrib.image.python.ops import interpolate_spline

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

from utils import dense_image_warp


def _get_grid_locations(image_depth, image_height, image_width):
    """Wrapper for np.meshgrid."""

    z_range = np.linspace(0, image_depth - 1, image_depth)
    y_range = np.linspace(0, image_height - 1, image_height)
    x_range = np.linspace(0, image_width - 1, image_width)
    z_grid, y_grid, x_grid = np.meshgrid(
        z_range, y_range, x_range, indexing='ij')

    return np.stack((z_grid, y_grid, x_grid), -1)


def _expand_to_minibatch(np_array, batch_size):
    """Tile arbitrarily-sized np_array to include new batch dimension."""
    tiles = [batch_size] + [1] * np_array.ndim
    return np.tile(np.expand_dims(np_array, 0), tiles)


def _get_boundary_locations(image_depth, image_height, image_width,
                            num_points_per_edge):
    """Compute evenly-spaced indices along edge of image."""
    z_range = np.linspace(0, image_depth - 1, num_points_per_edge + 2)
    y_range = np.linspace(0, image_height - 1, num_points_per_edge + 2)
    x_range = np.linspace(0, image_width - 1, num_points_per_edge + 2)
    zs, ys, xs = np.meshgrid(z_range, y_range, x_range, indexing='ij')
    is_boundary = np.logical_or(
        np.logical_or(xs == 0, xs == image_width - 1),
        np.logical_or(ys == 0, ys == image_height - 1),
        np.logical_or(zs == 0, zs == image_depth - 1))
    return np.stack([zs[is_boundary], ys[is_boundary], xs[is_boundary]], axis=-1)


def _add_zero_flow_controls_at_boundary(control_point_locations,
                                        control_point_flows, image_depth,
                                        image_height,
                                        image_width, boundary_points_per_edge):
    """Add control points for zero-flow boundary conditions.
     Augment the set of control points with extra points on the
     boundary of the image that have zero flow.
    Args:
      control_point_locations: input control points
      control_point_flows: their flows
      image_depth: image depth
      image_height: image height
      image_width: image width
      boundary_points_per_edge: number of points to add in the middle of each
                             edge (not including the corners).
                             The total number of points added is
                             8 + 12*(boundary_points_per_edge).
    Returns:
      merged_control_point_locations: augmented set of control point locations
      merged_control_point_flows: augmented set of control point flows
    """

    batch_size = tensor_shape.dimension_value(control_point_locations.shape[0])

    boundary_point_locations = _get_boundary_locations(image_depth, image_height,
                                                       image_width,
                                                       boundary_points_per_edge)

    boundary_point_flows = np.zeros([boundary_point_locations.shape[0], 3])

    type_to_use = control_point_locations.dtype
    boundary_point_locations = constant_op.constant(
        _expand_to_minibatch(boundary_point_locations, batch_size),
        dtype=type_to_use)

    boundary_point_flows = constant_op.constant(
        _expand_to_minibatch(boundary_point_flows, batch_size),
        dtype=type_to_use)

    merged_control_point_locations = array_ops.concat(
        [control_point_locations, boundary_point_locations], 1)

    merged_control_point_flows = array_ops.concat(
        [control_point_flows, boundary_point_flows], 1)

    return merged_control_point_locations, merged_control_point_flows


def sparse_image_warp(image,
                      source_control_point_locations,
                      dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundary_points=0,
                      name='sparse_image_warp'):
    dims = source_control_point_locations.get_shape().as_list()[2]
    if dims == 3:
        return sparse_image_warp_3d(
            image=image,
            source_control_point_locations=source_control_point_locations,
            dest_control_point_locations=dest_control_point_locations,
            interpolation_order=interpolation_order,
            regularization_weight=regularization_weight,
            num_boundary_points=num_boundary_points,
            name=name,
        )
    elif dims == 2:
        return tf.contrib.image.sparse_image_warp(
            image,
            source_control_point_locations=source_control_point_locations,
            dest_control_point_locations=dest_control_point_locations,
            interpolation_order=interpolation_order,
            regularization_weight=regularization_weight,
            num_boundary_points=num_boundary_points,
            name=name,
        )
    else:
        raise NotImplementedError


def sparse_image_warp_3d(image,
                         source_control_point_locations,
                         dest_control_point_locations,
                         interpolation_order=2,
                         regularization_weight=0.0,
                         num_boundary_points=0,
                         name='sparse_image_warp'):
    """Image warping using correspondences between sparse control points.
    Apply a non-linear warp to the image, where the warp is specified by
    the source and destination locations of a (potentially small) number of
    control points. First, we use a polyharmonic spline
    (`tf.contrib.image.interpolate_spline`) to interpolate the displacements
    between the corresponding control points to a dense flow field.
    Then, we warp the image using this dense flow field
    (`tf.contrib.image.dense_image_warp`).
    Let t index our control points. For regularization_weight=0, we have:
    warped_image[b, dest_control_point_locations[b, t, 0],
                    dest_control_point_locations[b, t, 1],
                    dest_control_point_locations[b, t, 2], :] =
    image[b, source_control_point_locations[b, t, 0],
             source_control_point_locations[b, t, 1],
             source_control_point_locations[b, t, 2], :].
    For regularization_weight > 0, this condition is met approximately, since
    regularized interpolation trades off smoothness of the interpolant vs.
    reconstruction of the interpolant at the control points.
    See `tf.contrib.image.interpolate_spline` for further documentation of the
    interpolation_order and regularization_weight arguments.
    Args:
      image: `[batch, depth, height, width, channels]` float `Tensor`
      source_control_point_locations: `[batch, num_control_points, 3]` float
        `Tensor`
      dest_control_point_locations: `[batch, num_control_points, 3]` float
        `Tensor`
      interpolation_order: polynomial order used by the spline interpolation
      regularization_weight: weight on smoothness regularizer in interpolation
      num_boundary_points: How many zero-flow boundary points to include at
        each image edge.Usage:
          num_boundary_points=0: don't add zero-flow points
          num_boundary_points=1: 8 corners of the image
          num_boundary_points=2: 8 corners and one in the middle of each edge
            (20 points total)
          num_boundary_points=n: 4 corners and n-1 along each edge
      name: A name for the operation (optional).
      Note that image and offsets can be of type tf.half, tf.float32, or
      tf.float64, and do not necessarily have to be the same type.
    Returns:
      warped_image: `[batch, depth, height, width, channels]` float `Tensor` with same
        type as input image.
      flow_field: `[batch, depth, height, width, 2]` float `Tensor` containing the dense
        flow field produced by the interpolation.
    """

    image = ops.convert_to_tensor(image)
    source_control_point_locations = ops.convert_to_tensor(
        source_control_point_locations)
    dest_control_point_locations = ops.convert_to_tensor(
        dest_control_point_locations)

    control_point_flows = dest_control_point_locations - source_control_point_locations

    clamp_boundaries = num_boundary_points > 0
    boundary_points_per_edge = num_boundary_points - 1

    with ops.name_scope(name):
        batch_size, image_depth, image_height, image_width, _ = image.get_shape().as_list()

        # This generates the dense locations where the interpolant will be evaluated.
        grid_locations = _get_grid_locations(
            image_depth, image_height, image_width)

        flattened_grid_locations = np.reshape(
            grid_locations,
            [image_depth * image_height * image_width, 3],
        )

        flattened_grid_locations = constant_op.constant(
            _expand_to_minibatch(flattened_grid_locations, batch_size),
            image.dtype,
        )

        if clamp_boundaries:
            dest_control_point_locations, control_point_flows = _add_zero_flow_controls_at_boundary(
                dest_control_point_locations,
                control_point_flows,
                image_depth,
                image_height,
                image_width,
                boundary_points_per_edge,
            )

        flattened_flows = interpolate_spline.interpolate_spline(
            dest_control_point_locations,
            control_point_flows,
            flattened_grid_locations,
            interpolation_order,
            regularization_weight,
        )

        dense_flows = array_ops.reshape(
            flattened_flows,
            [batch_size, image_depth, image_height, image_width, 3],
        )

        warped_image = dense_image_warp.dense_image_warp(image, dense_flows)

        return warped_image, dense_flows
