import numpy as np
import tensorflow as tf

from metrics import confusion_metrics
from metrics import pc_metrics
from dataset_helpers import prostate_cancer_utils


def get_metrics(prediction_groundtruth_stack,
                parallel_iterations, calc_froc, is_3d, thresholds):
  prediction, groundtruth = tf.unstack(prediction_groundtruth_stack, axis=0)
  groundtruth = tf.cast(groundtruth, tf.int64)

  if is_3d:
    assert(len(prediction.get_shape()) == 3)
    assert(len(groundtruth.get_shape()) == 3)
  else:
    assert(len(prediction.get_shape()) == 2)
    assert(len(groundtruth.get_shape()) == 2)

  prediction = tf.cast(tf.greater_equal(prediction, 0.5), tf.int64)

  split_groundtruth = prostate_cancer_utils.split_mask(
      groundtruth, dilate_mask=False, is_3d=is_3d)
  num_lesions = tf.shape(split_groundtruth, out_type=tf.int64)[0]

  tp = confusion_metrics.true_positives(labels=groundtruth,
                                        predictions=prediction)
  fp = confusion_metrics.false_positives(labels=groundtruth,
                                         predictions=prediction)
  fn = confusion_metrics.false_negatives(labels=groundtruth,
                                         predictions=prediction)

  region_cm_values = pc_metrics.get_region_cm_values(
    prediction=prediction, split_groundtruth=split_groundtruth,
    parallel_iterations=parallel_iterations, is_3d=is_3d)

  if calc_froc:
    # Get FROC values
    froc_region_cm_values = pc_metrics.get_region_cm_values_at_thresholds(
      prediction=prediction, split_groundtruth=split_groundtruth,
      thresholds=thresholds, parallel_iterations=parallel_iterations,
      is_3d=is_3d)
  else:
    froc_region_cm_values = {
      'region_tp': tf.constant(0, dtype=tf.int64),
      'region_fp': tf.constant(0, dtype=tf.int64),
      'region_fn': tf.constant(0, dtype=tf.int64)}

  metric_dict = {}
  # We want to collect the statistics for the regions so that we can calculate
  # patient centered metrics later
  statistics_dict = {}
  statistics_dict['region_tp'] = region_cm_values[0]
  statistics_dict['region_fn'] = region_cm_values[1]
  statistics_dict['region_fp'] = region_cm_values[2]
  statistics_dict['tp'] = tp
  statistics_dict['fn'] = fn
  statistics_dict['fp'] = fp

  return (metric_dict, statistics_dict, num_lesions, froc_region_cm_values)
