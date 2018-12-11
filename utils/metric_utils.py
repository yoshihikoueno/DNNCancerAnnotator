import functools

import numpy as np
import tensorflow as tf

from utils import util_ops
from metrics import metrics


def _split_groundtruth_mask(groundtruth):
  assert(len(groundtruth.get_shape()) == 2)

  # Label each cancer area with individual index
  components = tf.contrib.image.connected_components(groundtruth)

  unique_ids, unique_indices = tf.unique(tf.reshape(components, [-1]))

  # Remove zero id, since it describes background
  unique_ids = tf.gather_nd(unique_ids, tf.where(tf.not_equal(unique_ids, 0)))

  # Create mask for each cancer area
  individual_masks = tf.map_fn(
    lambda unique_id: tf.equal(unique_id, components), elems=unique_ids,
    dtype=tf.bool, parallel_iterations=4)

  return individual_masks


def _compute_region_recall_for_threshold(threshold, prediction,
                                         groundtruth_masks):
  assert(len(prediction.get_shape()) == 2)
  assert(len(groundtruth_masks.get_shape()) == 3)

  prediction = tf.greater(prediction, threshold)

  def _get_overlap(groundtruth, prediction):
    overlap_sum = tf.reduce_sum(
      tf.to_float(tf.logical_and(groundtruth, prediction)))
    return tf.div(overlap_sum, tf.reduce_sum(tf.to_float(groundtruth)))

  overlaps = tf.map_fn(lambda groundtruth: _get_overlap(
    groundtruth=groundtruth, prediction=prediction), elems=groundtruth_masks,
                       dtype=tf.float32, parallel_iterations=4)

  detections = tf.greater_equal(overlaps, tf.constant(0.3))

  num_tp = tf.reduce_sum(tf.cast(detections, tf.int32), axis=0,
                         name='num_tp_op')
  num_fn = tf.subtract(tf.size(detections), num_tp, name='num_fn_op')

  tps_count_op = tf.get_variable(
    'region_true_positives_{}'.format(
      int(threshold * 100)), dtype=tf.float32,
    collections=[tf.GraphKeys.LOCAL_VARIABLES],
    initializer=lambda: tf.constant(0, dtype=tf.float32),
    trainable=False)
  fns_count_op = tf.get_variable(
    'region_false_negatives_{}'.format(
      int(threshold * 100)), dtype=tf.float32,
    collections=[tf.GraphKeys.LOCAL_VARIABLES],
    initializer=lambda: tf.constant(0, dtype=tf.float32),
    trainable=False)

  # This identity is necessary, since the assign_add below works on a reference
  # and we want to keep the returned variable independent of the update
  tps_count_var = tf.identity(tps_count_op)
  fns_count_var = tf.identity(fns_count_op)

  tps_update_op = tf.assign_add(tps_count_op, tf.to_float(num_tp),
                                use_locking=True)
  fns_update_op = tf.assign_add(fns_count_op, tf.to_float(num_fn),
                                use_locking=True)

  def compute_recall(true_p, false_n, name):
    return tf.where(tf.greater(true_p + false_n, 0),
                    tf.div(true_p, true_p + false_n), 0, name=name)

  recall = compute_recall(tps_count_var, fns_count_var,
                          name='recall_op')

  recall_update_op = compute_recall(tps_update_op, fns_update_op,
                                    name='recall_update_op')

  return recall, recall_update_op, num_tp, num_fn


# Prediction and  groundtruth both NxHxW
def _compute_region_recall(prediction_groundtruth_stack, tp_thresholds):
  prediction, groundtruth = tf.unstack(prediction_groundtruth_stack, axis=0)
  assert(len(prediction.get_shape()) == 2)
  assert(len(groundtruth.get_shape()) == 2)

  individual_masks = _split_groundtruth_mask(groundtruth)

  fn = functools.partial(
    _compute_region_recall_for_threshold, prediction=prediction,
    groundtruth_masks=individual_masks)

  return list(map(fn, tp_thresholds))


def get_metrics(prediction_batch, groundtruth_batch, tp_thresholds,
                batch_size):
  assert(len(prediction_batch.get_shape()) == 4)
  assert(len(groundtruth_batch.get_shape()) == 4)

  # Get values between 0 and 1
  prediction_batch = tf.nn.softmax(prediction_batch)[:, :, :, 1]
  groundtruth_batch = tf.squeeze(groundtruth_batch, axis=3)

  precision = tf.metrics.precision_at_thresholds(
    labels=groundtruth_batch, predictions=prediction_batch,
    thresholds=tp_thresholds)
  recall = tf.metrics.recall_at_thresholds(
    labels=groundtruth_batch, predictions=prediction_batch,
    thresholds=tp_thresholds)

  auc = tf.metrics.auc(groundtruth_batch, prediction_batch)

 # region_recall_fn = functools.partial(_compute_region_recall,
  #                                     tp_thresholds=tp_thresholds)
  #region_recall = tf.map_fn(
  #  lambda prediction_groundtruth_stack: region_recall_fn(
  #    prediction_groundtruth_stack), elems=tf.stack(
  #      [prediction_batch, groundtruth_batch], axis=1),
  #  dtype=[(tf.float32, tf.float32, tf.int32, tf.int32)] * len(tp_thresholds),
  #  parallel_iterations=min(batch_size, util_ops.get_cpu_count()))

  region_recall = metrics.region_recall_at_thresholds(
    groundtruth_batch, prediction_batch, tp_thresholds)

  metric_dict = {'metrics/auc': auc}

  # We want to collect the statistics for the regions so that we can calculate
  # patient centered metrics later
  region_statistics_dict = {}
  for i, t in enumerate(tp_thresholds):
    t = int(np.round(t * 100))
    metric_dict['metrics/precision_at_{}'.format(t)] = (
      precision[0][i], precision[1][i])
    metric_dict['metrics/recall_at_{}'.format(t)] = (
      recall[0][i], recall[1][i])

    metric_dict['metrics/region_recall_at_{}'.format(t)] = (
      region_recall[i][0], region_recall[i][1])

    region_statistics_dict[t] = (
      region_recall[i][2], region_recall[i][3])

  return metric_dict, region_statistics_dict
