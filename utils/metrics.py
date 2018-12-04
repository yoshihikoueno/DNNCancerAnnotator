import numpy as np
import tensorflow as tf


def get_metrics(prediction_batch, groundtruth_batch, thresholds):
  assert(len(prediction_batch.get_shape()) == 4)
  assert(len(groundtruth_batch.get_shape()) == 4)

  # Get values between 0 and 1
  prediction_batch = tf.nn.softmax(prediction_batch)[:, :, :, 1]

  precision = tf.metrics.precision_at_thresholds(
    labels=groundtruth_batch, predictions=prediction_batch,
    thresholds=thresholds)
  recall = tf.metrics.recall_at_thresholds(
    labels=groundtruth_batch, predictions=prediction_batch,
    thresholds=thresholds)

  auc = tf.metrics.auc(groundtruth_batch, prediction_batch)

  components = tf.contrib.image.connected_components(tf.squeeze(
    groundtruth_batch, axis=3))
  unique_ids, unique_indices = tf.unique(components)

  # Remove zero id
  unique_ids = tf.gather_nd(unique_ids, tf.where(tf.not_equal(unique_ids, 0)))

  equal_fn = functools.partial(tf.equal, x=components)
  individual_masks = tf.map_fn(equal_fn, elems=unique_ids,
                               parallel_iterations=4)



  metric_dict = {'Metrics/auc': auc}
  for i, t in enumerate(thresholds):
    metric_dict['Metrics/precision_at_{}'.format(
      int(np.round(t * 100)))] = (precision[0][i], precision[1][i])
    metric_dict['Metrics/recall_at_{}'.format(
      int(np.round(t * 100)))] = (recall[0][i], recall[1][i])

  return metric_dict
