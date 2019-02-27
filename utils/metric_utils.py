import numpy as np
import tensorflow as tf

from metrics import region_recall_metric
from metrics import confusion_metrics
from utils import util_ops


def get_metrics(prediction_batch, groundtruth_batch, tp_thresholds,
                parallel_iterations):
  assert(len(prediction_batch.get_shape()) == 3)
  assert(len(groundtruth_batch.get_shape()) == 4)

  groundtruth_batch = tf.squeeze(groundtruth_batch, axis=3)

  tp_batch = tf.map_fn(
    lambda pred_gt_stack: confusion_metrics.true_positives_at_thresholds(
      labels=tf.cast(pred_gt_stack[1], dtype=tf.int64),
      predictions=pred_gt_stack[0],
      thresholds=tp_thresholds), elems=tf.stack([
        prediction_batch, tf.to_float(groundtruth_batch)], axis=1),
    parallel_iterations=util_ops.get_cpu_count(),
    dtype=tf.int64)

  fp_batch = tf.map_fn(
    lambda pred_gt_stack: confusion_metrics.false_positives_at_thresholds(
      labels=tf.cast(pred_gt_stack[1], dtype=tf.int64),
      predictions=pred_gt_stack[0],
      thresholds=tp_thresholds), elems=tf.stack([
        prediction_batch, tf.to_float(groundtruth_batch)], axis=1),
    parallel_iterations=util_ops.get_cpu_count(),
    dtype=tf.int64)

  fn_batch = tf.map_fn(
    lambda pred_gt_stack: confusion_metrics.false_negatives_at_thresholds(
      labels=tf.cast(pred_gt_stack[1], dtype=tf.int64),
      predictions=pred_gt_stack[0],
      thresholds=tp_thresholds), elems=tf.stack([
        prediction_batch, tf.to_float(groundtruth_batch)], axis=1),
    parallel_iterations=util_ops.get_cpu_count(),
    dtype=tf.int64)

  tn_batch = tf.map_fn(
    lambda pred_gt_stack: confusion_metrics.true_negatives_at_thresholds(
      labels=tf.cast(pred_gt_stack[1], dtype=tf.int64),
      predictions=pred_gt_stack[0],
      thresholds=tp_thresholds), elems=tf.stack([
        prediction_batch, tf.to_float(groundtruth_batch)], axis=1),
    parallel_iterations=util_ops.get_cpu_count(),
    dtype=tf.int64)

  # precision = tf.metrics.precision_at_thresholds(
  #   labels=groundtruth_batch, predictions=prediction_batch,
  #   thresholds=tp_thresholds)

  # recall = tf.metrics.recall_at_thresholds(
  #   labels=groundtruth_batch, predictions=prediction_batch,
  #   thresholds=tp_thresholds)

  # auc = tf.metrics.auc(groundtruth_batch, prediction_batch)

  # f1_score = tf.where(tf.greater(1 * precision[0] + recall[0], 0.0),
  #                     (2 * precision[0] * recall[0]) / (
  #                       1 * precision[0] + recall[0]), [0.0]
  #                     * len(tp_thresholds))
  # with tf.control_dependencies([precision[1], recall[1]]):
  #   f1_score_update_op = tf.identity(f1_score)

  # f1_5_score = tf.where(tf.greater(2.25 * precision[0] + recall[0], 0.0),
  #                       (3.25 * precision[0] * recall[0]) / (
  #                         2.25 * precision[0] + recall[0]), [0.0]
  #                       * len(tp_thresholds))
  # with tf.control_dependencies([precision[1], recall[1]]):
  #   f1_5_score_update_op = tf.identity(f1_5_score)

  # # More weight on recall
  # f2_score = tf.where(tf.greater(4 * precision[0] + recall[0], 0.0),
  #                     (5 * precision[0] * recall[0]) / (
  #                       4 * precision[0] + recall[0]), [0.0]
  #                     * len(tp_thresholds))
  # with tf.control_dependencies([precision[1], recall[1]]):
  #   f2_score_update_op = tf.identity(f2_score)

  region_recall = region_recall_metric.region_recall_at_thresholds(
    groundtruth_batch, prediction_batch, tp_thresholds,
    parallel_iterations=parallel_iterations)

  #metric_dict = {'metrics/auc': auc}
  metric_dict = {}

  # We want to collect the statistics for the regions so that we can calculate
  # patient centered metrics later
  statistics_dict = {}
  for i, t in enumerate(tp_thresholds):
    t = int(np.round(t * 100))
    # metric_dict['metrics/precision_at_{}'.format(t)] = (
    #   precision[0][i], precision[1][i])
    # metric_dict['metrics/recall_at_{}'.format(t)] = (
    #   recall[0][i], recall[1][i])
    # metric_dict['metrics/f1_score_at_{}'.format(t)] = (
    #   f1_score[i], f1_score_update_op[i])
    # metric_dict['metrics/f1_5_score_at_{}'.format(t)] = (
    #   f1_5_score[i], f1_5_score_update_op[i])
    # metric_dict['metrics/f2_score_at_{}'.format(t)] = (
    #   f2_score[i], f2_score_update_op[i])

    #metric_dict['metrics/region_recall_at_{}'.format(t)] = (
    #  region_recall[i][0], region_recall[i][1])

    assert(t not in statistics_dict)

    statistics_dict[t] = dict()

    statistics_dict[t]['region_tp'] = region_recall[i][2]
    statistics_dict[t]['region_fn'] = region_recall[i][3]
    statistics_dict[t]['tp'] = tp_batch[:, i]
    statistics_dict[t]['fp'] = fp_batch[:, i]
    statistics_dict[t]['fn'] = fn_batch[:, i]
    statistics_dict[t]['tn'] = tn_batch[:, i]

  return metric_dict, statistics_dict

def _get_region_based_metrics(prediction_batch, groundtruth_batch,
                              tp_thresholds, parallel_iterations):
  assert(len(prediction_batch.get_shape()) == 3)
  assert(len(groundtruth_batch.get_shape()) == 3)
