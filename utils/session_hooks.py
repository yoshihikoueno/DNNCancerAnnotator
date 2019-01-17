import logging
import os

import tensorflow as tf
import numpy as np

from utils import image_utils


class VisualizationHook(tf.train.SessionRunHook):
  def __init__(self, result_folder, visualization_file_names,
               file_name, image_decoded, annotation_decoded, annotation_mask,
               predicted_mask, eval_dir):
    assert(len(predicted_mask.get_shape()) == 3)
    self.visualization_file_names = visualization_file_names
    self.result_folder = result_folder
    self.file_name = file_name
    self.eval_dir = eval_dir

    target_size = predicted_mask.get_shape().as_list()[1:3]

    image_decoded = image_utils.central_crop(image_decoded, target_size)
    image_decoded = tf.image.grayscale_to_rgb(image_decoded)
    predicted_mask = tf.stack([predicted_mask * 255,
                               tf.zeros_like(predicted_mask),
                              tf.zeros_like(predicted_mask)], axis=3)

    annotation_decoded = image_utils.central_crop(
      annotation_decoded, target_size)

    predicted_mask_overlay = tf.clip_by_value(
      image_decoded * 0.5 + predicted_mask, 0, 255)

    self.combined_image = tf.concat([
      image_decoded, annotation_decoded, predicted_mask_overlay,
      predicted_mask], axis=2)

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=[
      self.file_name, self.combined_image, tf.train.get_global_step()])

  def after_run(self, run_context, run_values):
    file_name_res = run_values.results[0]
    combined_image_res = run_values.results[1]
    global_step = run_values.results[2]

    # Estimator writes summaries to the eval subfolder
    summary_writer = tf.summary.FileWriterCache.get(
      os.path.join(self.result_folder, self.eval_dir))

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
  def __init__(self, statistics_dict, patient_id, result_folder,
               tp_thresholds, eval_dir):
    self.statistics_dict = statistics_dict
    self.patient_id = patient_id
    self.result_folder = result_folder
    self.tp_thresholds = tp_thresholds
    self.eval_dir = eval_dir

    self.patient_statistics = dict()

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=[self.statistics_dict,
                                            self.patient_id])

  def after_run(self, run_context, run_values):
    statistics_dict_res = run_values.results[0]
    patient_id_res = run_values.results[1]

    for patient_id in patient_id_res:
      if patient_id not in self.patient_statistics:
        self.patient_statistics[patient_id] = dict()
        self.patient_statistics[patient_id]['num_slices'] = 0
        self.patient_statistics[patient_id]['thresholds'] = dict()

      self.patient_statistics[patient_id]['num_slices'] += 1

    for threshold, confusion_dict in statistics_dict_res.items():
      for confusion_key, batch_values in confusion_dict.items():
        for i, val in enumerate(batch_values):
          if threshold not in self.patient_statistics[patient_id_res[i]][
              'thresholds']:
            self.patient_statistics[patient_id_res[i]]['thresholds'][
              threshold] = dict()

          if confusion_key not in self.patient_statistics[patient_id_res[i]][
              'thresholds'][threshold]:
            self.patient_statistics[patient_id_res[i]]['thresholds'][
              threshold][confusion_key] = 0

          self.patient_statistics[patient_id_res[i]]['thresholds'][threshold][
            confusion_key] += val

  def end(self, session):
    # Collect all normalized confusion values
    population_region_tp = dict()
    population_region_fn = dict()
    population_tp = dict()
    population_fp = dict()
    population_fn = dict()
    population_tn = dict()

    for threshold in self.tp_thresholds:
      threshold = int(np.round(threshold * 100))
      population_region_tp[threshold] = 0
      population_region_fn[threshold] = 0
      population_tp[threshold] = 0
      population_fp[threshold] = 0
      population_fn[threshold] = 0
      population_tn[threshold] = 0

    summary_writer = tf.summary.FileWriterCache.get(
      os.path.join(self.result_folder, self.eval_dir))
    global_step = tf.train.get_global_step().eval(session=session)

    for patient_id, statistics in self.patient_statistics.items():
      num_slices = statistics['num_slices']
      assert(num_slices > 0)
      threshold_statistics = statistics['thresholds']
      for threshold, confusion_dict in threshold_statistics.items():
        population_region_tp[threshold] += (confusion_dict['region_tp']
                                            / float(num_slices))
        population_region_fn[threshold] += (confusion_dict['region_fn']
                                            / float(num_slices))
        population_tp[threshold] += (confusion_dict['tp'] / float(num_slices))
        population_fp[threshold] += (confusion_dict['fp'] / float(num_slices))
        population_fn[threshold] += (confusion_dict['fn'] / float(num_slices))
        population_tn[threshold] += (confusion_dict['tn'] / float(num_slices))

    for threshold in self.tp_thresholds:
      threshold = int(np.round(threshold * 100))
      region_tp = population_region_tp[threshold]
      region_fn = population_region_fn[threshold]
      tp = population_tp[threshold]
      fp = population_fp[threshold]
      fn = population_fn[threshold]
      tn = population_tn[threshold]

      recall = tp / (tp + fn) if (tp + fn > 0) else 0
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/population_recall_at_{}'.format(
          threshold), simple_value=recall)
      summary_writer.add_summary(summary, global_step=global_step)

      precision = tp / (tp + fp) if (tp + fp > 0) else 0
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/population_precision_at_{}'.format(
          threshold), simple_value=precision)
      summary_writer.add_summary(summary, global_step=global_step)

      region_recall = region_tp / (region_tp + region_fn) if (
        region_tp + region_fn > 0) else 0
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/population_region_recall_at_{}'.format(
          threshold), simple_value=region_recall)
      summary_writer.add_summary(summary, global_step=global_step)

      f1_score = (2 * precision * recall / (precision + recall)) if (
          precision + recall) > 0 else 0
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/population_f1_score_at_{}'.format(
          threshold), simple_value=f1_score)
      summary_writer.add_summary(summary, global_step=global_step)

      f2_score = (5 * precision * recall / (4 * precision + recall)) if (
        (4 * precision + recall)) > 0 else 0
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/population_f2_score_at_{}'.format(
          threshold), simple_value=f2_score)
      summary_writer.add_summary(summary, global_step=global_step)


class PrintHook(tf.train.SessionRunHook):
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=self.kwargs)

  def after_run(self, run_context, run_values):
    kwargs_res = run_values.results

    for key, value in kwargs_res.items():
      logging.info("{}: {}".format(key, value))
