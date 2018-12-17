import numpy as np
import tensorflow as tf

from metrics import region_recall_metric


def get_metrics(prediction_batch, groundtruth_batch, tp_thresholds,
                parallel_iterations):
  assert(len(prediction_batch.get_shape()) == 3)
  assert(len(groundtruth_batch.get_shape()) == 4)

  groundtruth_batch = tf.squeeze(groundtruth_batch, axis=3)

  precision = tf.metrics.precision_at_thresholds(
    labels=groundtruth_batch, predictions=prediction_batch,
    thresholds=tp_thresholds)
  recall = tf.metrics.recall_at_thresholds(
    labels=groundtruth_batch, predictions=prediction_batch,
    thresholds=tp_thresholds)

  auc = tf.metrics.auc(groundtruth_batch, prediction_batch)

  f1_score = tf.where(tf.greater(1 * precision[0] + recall[0], 0.0),
                      (2 * precision[0] * recall[0]) / (
                        1 * precision[0] * recall[0]), [0.0]
                      * len(tp_thresholds))
  with tf.control_dependencies([precision[1], recall[1]]):
    f1_score_update_op = tf.identity(f1_score)

  # More weight on recall
  f2_score = tf.where(tf.greater(4 * precision[0] + recall[0], 0.0),
                      (5 * precision[0] * recall[0]) / (
                        4 * precision[0] * recall[0]), [0.0]
                      * len(tp_thresholds))
  with tf.control_dependencies([precision[1], recall[1]]):
    f2_score_update_op = tf.identity(f2_score)

  region_recall = region_recall_metric.region_recall_at_thresholds(
    groundtruth_batch, prediction_batch, tp_thresholds,
    parallel_iterations=parallel_iterations)

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
    metric_dict['metrics/f1_score_at_{}'.format(t)] = (
      f1_score[i], f1_score_update_op[i])
    metric_dict['metrics/f2_score_at_{}'.format(t)] = (
      f2_score[i], f2_score_update_op[i])

    metric_dict['metrics/region_recall_at_{}'.format(t)] = (
      region_recall[i][0], region_recall[i][1])

    region_statistics_dict[t] = (
      region_recall[i][2], region_recall[i][3])

  return metric_dict, region_statistics_dict
