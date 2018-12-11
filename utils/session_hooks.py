import logging

import tensorflow as tf
import numpy as np

from utils import standard_fields
from utils import image_utils


class VisualizationHook(tf.train.SessionRunHook):
  def __init__(self, result_folder, visualization_file_names,
               file_name, image_decoded, annotation_decoded, annotation_mask,
               predicted_masks):
    self.visualization_file_names = visualization_file_names
    self.result_folder = result_folder
    self.file_name = file_name

    background, foreground = tf.split(tf.nn.softmax(predicted_masks), 2,
                                      axis=3)

    target_size = background.get_shape().as_list()[1:3]

    image_decoded = image_utils.central_crop(image_decoded, target_size)

    annotation_decoded = image_utils.central_crop(
      annotation_decoded, target_size)

    background = tf.image.grayscale_to_rgb(background * 255)
    foreground = tf.image.grayscale_to_rgb(foreground * 255)
    image_decoded = tf.image.grayscale_to_rgb(image_decoded)
    annotation_mask = tf.image.grayscale_to_rgb(annotation_mask * 255)

    self.combined_image = tf.concat([
      image_decoded, annotation_decoded, annotation_mask, background,
      foreground], axis=2)

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=[
      self.file_name, self.combined_image, tf.train.get_global_step()])

  def after_run(self, run_context, run_values):
    file_name_res = run_values.results[0]
    combined_image_res = run_values.results[1]
    global_step = run_values.results[2]

    summary_writer = tf.summary.FileWriterCache.get(self.result_folder)

    for batch_index in range(len(combined_image_res)):
      file_name = file_name_res[batch_index].decode('utf-8')

      if file_name in self.visualization_file_names:
        summary = tf.Summary(value=[
          tf.Summary.Value(
            tag=file_name,
            image=tf.Summary.Image(
              encoded_image_string=image_utils.encode_image_array_as_png_str(
                combined_image_res[batch_index])))])
        summary_writer.add_summary(summary, global_step)

        logging.info('Visualization for {}'.format(file_name))
      else:
        logging.info('Skipping visualization for {}'.format(file_name))
        continue


class PatientMetricHook(tf.train.SessionRunHook):
  def __init__(self, region_statistics_dict, patient_id, result_folder):
    self.region_statistics_dict = region_statistics_dict
    self.patient_id = patient_id
    self.result_folder = result_folder

    self.patient_statistics = dict()

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=[self.region_statistics_dict,
                                            self.patient_id])

  def after_run(self, run_context, run_values):
    region_statistics_dict_res = run_values.results[0]
    patient_id_res = run_values.results[1]

    for threshold, tp_fn_batch in region_statistics_dict_res.items():
      # Since tp_fn still contains a batch dimension, we need to have a loop
      for i, tp_fn in enumerate(list(zip(*tp_fn_batch))):
        tp = tp_fn[0]
        fn = tp_fn[1]
        print('TP Threshold={}: TP: {}; FN: {}'.format(
          threshold, tp, fn))
        # If the current image does not contain any groundtruth, i.e.
        # if both tp and fn are 0, then we dont want to include it in our
        # Recall calculation later
        if tp == 0 and fn == 0:
          continue

        if patient_id_res[i] not in self.patient_statistics:
          self.patient_statistics[patient_id_res[i]] = dict()

        if threshold not in self.patient_statistics[patient_id_res[i]]:
          self.patient_statistics[patient_id_res[i]][threshold] = dict()
          self.patient_statistics[patient_id_res[i]][threshold]['tp'] = 0
          self.patient_statistics[patient_id_res[i]][threshold]['fn'] = 0

        self.patient_statistics[patient_id_res[i]][threshold]['tp'] += tp
        self.patient_statistics[patient_id_res[i]][threshold]['fn'] += fn

  def end(self, session):
    patient_recalls = dict()
    for _, statistics in self.patient_statistics.items():
      for threshold, tp_fn_dict in statistics.items():
        tp = tp_fn_dict['tp']
        fn = tp_fn_dict['fn']

        assert(tp + fn > 0)

        if threshold not in patient_recalls:
          patient_recalls[threshold] = []

        patient_recalls[threshold].append(tp / float((tp + fn)))

    summary_writer = tf.summary.FileWriterCache.get(self.result_folder)
    for threshold, recalls in patient_recalls.items():
      recall = np.mean(recalls)

      summary = tf.Summary()
      summary.value.add(tag='patient_adjusted_recall_at_{}'.format(threshold),
                        simple_value=recall)
      summary_writer.add_summary(summary)

      logging.info("Summary for patient adjusted recall@{}".format(threshold))
