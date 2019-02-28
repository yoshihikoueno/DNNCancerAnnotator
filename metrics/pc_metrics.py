import tensorflow as tf

from dataset_helpers import prostate_cancer_utils


def get_region_cm_values_at_thresholds(prediction_batch, groundtruth_batch,
                                       thresholds, parallel_iterations):
  # Currently only support batch size of 1
  prediction = tf.squeeze(prediction_batch, axis=0)
  groundtruth = tf.squeeze(groundtruth_batch, axis=0)
  assert(len(prediction.get_shape()) == 2)
  assert(len(groundtruth.get_shape()) == 2)

  split_groundtruth = prostate_cancer_utils.split_mask(groundtruth)

  result = []
  for threshold in thresholds:
    result.append(get_region_cm_values(
      tf.cast(tf.greater_equal(prediction, threshold), tf.int64),
      split_groundtruth=split_groundtruth,
      parallel_iterations=parallel_iterations))

  return result


def get_region_cm_values(prediction, split_groundtruth, parallel_iterations):
  assert(len(prediction.get_shape()) == 2)
  assert(len(split_groundtruth.get_shape()) == 3)
  assert(prediction.dtype == tf.int64)

  split_prediction = prostate_cancer_utils.split_mask(prediction,
                                                      dilate_mask=True)

  def calc_without_groundtruth():
    return (tf.expand_dims(tf.constant(0, dtype=tf.int64), axis=0),
            tf.expand_dims(tf.constant(0, dtype=tf.int64), axis=0),
            tf.expand_dims(tf.shape(split_prediction)[0], axis=0))

  def calc_with_groundtruth():
    # For every groundtruth mask, we want to assign the closest prediction if
    # f score is larger than a certain threshold
    # Every groundtruth can only have one prediction assigned
    i0 = tf.constant(1)
    f_scores0 = tf.expand_dims(tf.map_fn(
      lambda prediction: _get_f_score(
        prediction, split_groundtruth[0]), elems=split_prediction,
      parallel_iterations=parallel_iterations,
      dtype=tf.float32), axis=0)

    def while_cond(i, f_scores):
      return tf.less(i, tf.shape(split_groundtruth)[0])

    def while_body(i, f_scores):
      f_score = tf.map_fn(lambda prediction: _get_f_score(
        prediction, split_groundtruth[i]), elems=split_prediction,
                              parallel_iterations=parallel_iterations,
                              dtype=tf.float32)
      return tf.add(i, 1), tf.concat([f_scores, tf.expand_dims(
        f_score, axis=0)], axis=0)

    f_scores = tf.while_loop(while_cond, while_body, loop_vars=[i0, f_scores0],
                             shape_invariants=[i0.get_shape(), tf.TensorShape(
                               [None, None])])[1]

    best_f_scores = tf.reduce_max(f_scores, axis=1)
    best_predictions = tf.argmax(f_scores, axis=1)

    assigned_predictions = tf.greater(best_f_scores, 0.3)

    region_tp = tf.reduce_sum(tf.cast(assigned_predictions, tf.int64))
    region_fn = tf.reduce_sum(tf.cast(tf.logical_not(assigned_predictions),
                                      tf.int64))
    region_fp = tf.subtract(tf.shape(split_prediction)[0],
                            tf.shape(tf.unique(best_predictions)[0])[0])

    return (tf.expand_dims(region_tp, axis=0),
            tf.expand_dims(region_fn, axis=0),
            tf.expand_dims(region_fp, axis=0))

  return tf.cond(tf.equal(tf.shape(split_groundtruth)[0], tf.constant(0)),
                 calc_without_groundtruth, calc_with_groundtruth)


def _get_f_score(prediction, groundtruth):
  prediction = tf.cast(prediction, tf.bool)
  groundtruth = tf.cast(groundtruth, tf.bool)

  tp = tf.reduce_sum(tf.cast(tf.logical_and(prediction, groundtruth),
                             tf.float32))
  fp = tf.reduce_sum(tf.cast(tf.logical_and(prediction, tf.logical_not(
    groundtruth)), tf.float32))
  fn = tf.reduce_sum(tf.cast(tf.logical_and(
    tf.logical_not(prediction), groundtruth), tf.float32))

  gt_assert = tf.Assert(tf.greater((2 * tp + fp + fn), 0), [
    tf.constant('Groundtruth cannot be empty!')])
  with tf.control_dependencies([gt_assert]):
    return (2 * tp) / (2 * tp + fp + fn)
