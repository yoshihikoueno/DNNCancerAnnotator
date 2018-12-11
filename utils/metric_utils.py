import functools

import numpy as np
import tensorflow as tf

from metrics import metrics


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

  region_recall = metrics.region_recall_at_thresholds(
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

    metric_dict['metrics/region_recall_at_{}'.format(t)] = (
      region_recall[i][0], region_recall[i][1])

    region_statistics_dict[t] = (
      region_recall[i][2], region_recall[i][3])

  return metric_dict, region_statistics_dict
