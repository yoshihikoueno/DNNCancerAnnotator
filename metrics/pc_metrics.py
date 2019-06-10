import tensorflow as tf

from dataset_helpers import prostate_cancer_utils


def get_region_cm_values_at_thresholds(prediction, split_groundtruth,
                                       thresholds, parallel_iterations):
  assert(len(prediction.get_shape()) == 2)
  assert(len(split_groundtruth.get_shape()) == 3)

  v = tf.map_fn(lambda threshold: get_region_cm_values(
    tf.cast(tf.greater_equal(
      tf.cast(prediction, tf.float64), threshold), tf.int64),
    split_groundtruth=split_groundtruth,
    parallel_iterations=parallel_iterations), elems=thresholds,
                parallel_iterations=parallel_iterations,
                dtype=(tf.int64, tf.int64, tf.int64))

  v_dict = {'region_tp': v[0], 'region_fn': v[1], 'region_fp': v[2]}

  return v_dict


def get_region_cm_values(prediction, split_groundtruth, parallel_iterations,
                         is_3d):
  if is_3d:
    assert(len(prediction.get_shape()) == 3)
    assert(len(split_groundtruth.get_shape()) == 4)
  else:
    assert(len(prediction.get_shape()) == 2)
    assert(len(split_groundtruth.get_shape()) == 3)

  assert(prediction.dtype == tf.int64)

  split_prediction = prostate_cancer_utils.split_mask(
    prediction, dilate_mask=True, is_3d=is_3d)

  def calc_without_groundtruth():
    return (tf.constant(0, dtype=tf.int64),
            tf.constant(0, dtype=tf.int64),
            tf.cast(tf.shape(split_prediction)[0],
                                 dtype=tf.int64))

  def calc_with_groundtruth():
    def calc_with_prediction():
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

      f_scores = tf.while_loop(while_cond, while_body, loop_vars=[i0,
                                                                  f_scores0],
                               shape_invariants=[i0.get_shape(),
                                                 tf.TensorShape(
                                 [None, None])])[1]

      best_f_scores = tf.reduce_max(f_scores, axis=1)

      assigned_predictions = tf.greater(best_f_scores, 0.3)

      region_tp = tf.reduce_sum(tf.cast(assigned_predictions, tf.int64))
      region_fn = tf.reduce_sum(tf.cast(tf.logical_not(assigned_predictions),
                                        tf.int64))
      region_fp = tf.cast(tf.subtract(
        tf.cast(tf.shape(split_prediction)[0], tf.int64), region_tp),
                          tf.int64)

      return (region_tp, region_fn, region_fp)

    def calc_without_prediction():
      return (tf.constant(0, dtype=tf.int64),
              tf.cast(tf.shape(split_groundtruth)[0],
                      dtype=tf.int64),
              tf.constant(0, dtype=tf.int64))

    return tf.cond(tf.equal(tf.shape(split_prediction)[0], tf.constant(0)),
                   calc_without_prediction, calc_with_prediction)

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
