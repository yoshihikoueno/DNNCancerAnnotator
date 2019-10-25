import tensorflow as tf
import numpy as np

from dataset_helpers import prostate_cancer_utils


# groundtruth_masks = [MxHxW]
# M = Number of masks within one batch
# prediction = [HxW]
def _calculate_overlap(groundtruth_prediction_tuple):
    groundtruth_masks, prediction = groundtruth_prediction_tuple

    assert len(groundtruth_masks.get_shape()) == 3
    assert len(prediction.get_shape()) == 2

    def _get_overlap(groundtruth, prediction):
        assert len(prediction.get_shape()) == 2
        assert len(groundtruth.get_shape()) == 2

        overlap_sum = tf.reduce_sum(
            tf.to_float(tf.logical_and(groundtruth, prediction)))

        return tf.div(overlap_sum, tf.reduce_sum(tf.to_float(groundtruth)))

    return tf.map_fn(lambda groundtruth: _get_overlap(groundtruth, prediction),
                     elems=groundtruth_masks, dtype=tf.float32,
                     parallel_iterations=4)


# labels = [NxMxHxW]
# predictions = [NxHxW], bool
def region_recall(labels, predictions, parallel_iterations):
    assert(len(labels.get_shape()) == 4)
    assert(len(predictions.get_shape()) == 3)
    assert(predictions.dtype == tf.bool)

    # [NxM] overlaps
    overlaps = tf.map_fn(_calculate_overlap, elems=(labels, predictions),
                         dtype=tf.float32,
                         parallel_iterations=parallel_iterations)
    assert(len(overlaps.get_shape()) == 2)

    tps = tf.greater_equal(overlaps, tf.constant(0.3))
    fns = tf.logical_not(tps)

    # [N] tps, fns
    num_tp = tf.reduce_sum(tf.cast(tps, dtype=tf.int32), axis=1,
                           name='num_tp_op')
    num_fn = tf.reduce_sum(tf.cast(fns, dtype=tf.int32), axis=1,
                           name='num_fn_op')

    # tps_var = tf.get_variable(
    #   'region_true_positives', dtype=tf.float32,
    #   collections=[tf.GraphKeys.LOCAL_VARIABLES],
    #   initializer=lambda: tf.constant(0, dtype=tf.float32),
    #   trainable=False)

    # fns_var = tf.get_variable(
    #   'region_false_negatives', dtype=tf.float32,
    #   collections=[tf.GraphKeys.LOCAL_VARIABLES],
    #   initializer=lambda: tf.constant(0, dtype=tf.float32),
    #   trainable=False)

    # tps_var_op = tf.identity(tps_var)
    # fns_var_op = tf.identity(fns_var)

    # tps_update_op = tf.assign_add(tps_var, tf.to_float(tf.reduce_sum(num_tp)),
    #                               use_locking=True)
    # fns_update_op = tf.assign_add(fns_var, tf.to_float(tf.reduce_sum(num_fn)),
    #                               use_locking=True)

    # def compute_recall(true_p, false_n, name):
    #   return tf.where(tf.greater(true_p + false_n, 0),
    #                   tf.div(true_p, true_p + false_n), 0, name=name)

    # recall = compute_recall(tps_var_op, fns_var_op, name='region_recall_op')
    # recall_update_op = compute_recall(tps_update_op, fns_update_op,
    #                                   name='region_recall_update_op')

    # return recall, recall_update_op, num_tp, num_fn
    return num_tp, num_fn


# labels = [NxHxW]
# predictions = [NxHxW]
def region_recall_at_thresholds(labels, predictions, thresholds,
                                parallel_iterations):
    assert(len(labels.get_shape()) == 3)
    assert(len(predictions.get_shape()) == 3)
    assert(predictions.dtype == tf.float32)

    # [NxMxHxW]
    # M = Number of masks
    # N = batch size
    groundtruth_masks = tf.map_fn(
        prostate_cancer_utils.split_mask, elems=labels, dtype=tf.bool,
        parallel_iterations=parallel_iterations)

    result = []
    for threshold in thresholds:
        with tf.variable_scope('TP_threshold_{}'.format(int(np.round(threshold
                                                                     * 100)))):
            result.append(region_recall(groundtruth_masks,
                                        tf.greater_equal(
                                            predictions, threshold),
                                        parallel_iterations))

    return result
