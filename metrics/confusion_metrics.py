import functools

import tensorflow as tf


def true_positives(predictions, labels):
  labels = tf.cast(labels, dtype=tf.bool)
  predictions = tf.cast(predictions, dtype=tf.bool)

  return tf.reduce_sum(tf.cast(tf.logical_and(labels, predictions),
                               dtype=tf.int64))


def true_negatives(predictions, labels):
  labels = tf.cast(labels, dtype=tf.bool)
  predictions = tf.cast(predictions, dtype=tf.bool)

  return tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(labels),
                                              tf.logical_not(predictions)),
                               dtype=tf.int64))


def false_negatives(predictions, labels):
  labels = tf.cast(labels, dtype=tf.bool)
  predictions = tf.cast(predictions, dtype=tf.bool)

  return tf.reduce_sum(tf.cast(tf.logical_and(labels,
                                              tf.logical_not(predictions)),
                               dtype=tf.int64))


def false_positives(predictions, labels):
  labels = tf.cast(labels, dtype=tf.bool)
  predictions = tf.cast(predictions, dtype=tf.bool)

  return tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(labels),
                                              predictions), dtype=tf.int64))


def true_positives_at_thresholds(labels, predictions, thresholds):
  prediction_masks = tf.map_fn(lambda threshold: tf.greater_equal(
    predictions, threshold), elems=thresholds, dtype=tf.bool)

  fn = functools.partial(true_positives, labels=labels)

  return tf.map_fn(fn, elems=prediction_masks, dtype=tf.int64)


def true_negatives_at_thresholds(labels, predictions, thresholds):
  prediction_masks = tf.map_fn(lambda threshold: tf.greater_equal(
    predictions, threshold), elems=thresholds, dtype=tf.bool)

  fn = functools.partial(true_negatives, labels=labels)

  return tf.map_fn(fn, elems=prediction_masks, dtype=tf.int64)


def false_negatives_at_thresholds(labels, predictions, thresholds):
  prediction_masks = tf.map_fn(lambda threshold: tf.greater_equal(
    predictions, threshold), elems=thresholds, dtype=tf.bool)

  fn = functools.partial(false_negatives, labels=labels)

  return tf.map_fn(fn, elems=prediction_masks, dtype=tf.int64)


def false_positives_at_thresholds(labels, predictions, thresholds):
  prediction_masks = tf.map_fn(lambda threshold: tf.greater_equal(
    predictions, threshold), elems=thresholds, dtype=tf.bool)

  fn = functools.partial(false_positives, labels=labels)

  return tf.map_fn(fn, elems=prediction_masks, dtype=tf.int64)
