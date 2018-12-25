import logging
import os

import tensorflow as tf
import numpy as np

from utils import image_utils


class VisualizationHook(tf.train.SessionRunHook):
  def __init__(self, result_folder, visualization_file_names,
               file_name, image_decoded, annotation_decoded, annotation_mask,
               predicted_mask):
    assert(len(predicted_mask.get_shape()) == 3)
    self.visualization_file_names = visualization_file_names
    self.result_folder = result_folder
    self.file_name = file_name

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
      os.path.join(self.result_folder, 'eval'))

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
               tp_thresholds):
    self.statistics_dict = statistics_dict
    self.patient_id = patient_id
    self.result_folder = result_folder
    self.tp_thresholds = tp_thresholds

    self.patient_statistics = dict()

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=[self.statistics_dict,
                                            self.patient_id])

  def after_run(self, run_context, run_values):
    statistics_dict_res = run_values.results[0]
    patient_id_res = run_values.results[1]

    for threshold, confusion_dict in statistics_dict_res.items():
      for confusion_key, batch_values in confusion_dict.items():
        for i, val in enumerate(batch_values):
          if patient_id_res[i] not in self.patient_statistics:
            self.patient_statistics[patient_id_res[i]] = dict()

          if threshold not in self.patient_statistics[patient_id_res[i]]:
            self.patient_statistics[patient_id_res[i]][threshold] = dict()

          if confusion_key not in self.patient_statistics[patient_id_res[i]][
              threshold]:
            self.patient_statistics[patient_id_res[i]][threshold][
              confusion_key] = 0

          self.patient_statistics[patient_id_res[i]][threshold][
            confusion_key] += val

  def end(self, session):
    patient_healthy_tnr = dict()
    patient_cancer_tnr = dict()
    patient_recall = dict()
    patient_precision = dict()
    patient_region_recall = dict()
    patient_f1_score = dict()
    patient_f2_score = dict()

    for threshold in self.tp_thresholds:
      threshold = int(np.round(threshold * 100))
      patient_healthy_tnr[threshold] = []
      patient_cancer_tnr[threshold] = []
      patient_recall[threshold] = []
      patient_precision[threshold] = []
      patient_region_recall[threshold] = []
      patient_f1_score[threshold] = []
      patient_f2_score[threshold] = []

    summary_writer = tf.summary.FileWriterCache.get(
      os.path.join(self.result_folder, 'eval'))
    global_step = tf.train.get_global_step().eval(session=session)
    # Calculate per patient metrics
    for patient_id, statistics in self.patient_statistics.items():
      for threshold, confusion_dict in statistics.items():
        region_tp = confusion_dict['region_tp']
        region_fn = confusion_dict['region_fn']
        tp = confusion_dict['tp']
        fp = confusion_dict['fp']
        fn = confusion_dict['fn']
        tn = confusion_dict['tn']

        if tp == 0 and fn == 0:
          # There is no true label in the image, only calculate tnr
          tnr = tn / (tn + fp)
          patient_healthy_tnr[threshold].append(tnr)
          summary = tf.Summary()
          summary.value.add(tag='patient_metrics/{}/tnr_at_{}'.format(
            patient_id.decode('utf-8'), threshold), simple_value=tnr)
          summary_writer.add_summary(summary, global_step=global_step)
        else:
          assert(tn + fp > 0)
          tnr = tn / (tn + fp)
          patient_cancer_tnr[threshold].append(tnr)
          summary = tf.Summary()
          summary.value.add(tag='patient_metrics/{}/tnr_at_{}'.format(
            patient_id.decode('utf-8'), threshold), simple_value=tnr)
          summary_writer.add_summary(summary, global_step=global_step)

          assert(tp + fn > 0)
          recall = tp / (tp + fn)
          patient_recall[threshold].append(recall)
          summary = tf.Summary()
          summary.value.add(tag='patient_metrics/{}/recall_at_{}'.format(
            patient_id.decode('utf-8'), threshold), simple_value=recall)
          summary_writer.add_summary(summary, global_step=global_step)

          precision = tp / (tp + fp) if (tp + fp) > 0 else 0
          patient_precision[threshold].append(precision)
          summary = tf.Summary()
          summary.value.add(tag='patient_metrics/{}/precision_at_{}'.format(
            patient_id.decode('utf-8'), threshold), simple_value=precision)
          summary_writer.add_summary(summary, global_step=global_step)

          assert(region_tp + region_fn > 0)
          region_recall = region_tp / (region_tp + region_fn)
          patient_region_recall[threshold].append(region_recall)
          summary = tf.Summary()
          summary.value.add(
            tag='patient_metrics/{}/region_recall_at_{}'.format(
              patient_id.decode('utf-8'), threshold),
            simple_value=region_recall)
          summary_writer.add_summary(summary, global_step=global_step)

          f1_score = (2 * precision * recall / (precision + recall)) if (
            precision + recall) > 0 else 0
          patient_f1_score[threshold].append(f1_score)
          summary = tf.Summary()
          summary.value.add(
            tag='patient_metrics/{}/f1_score_at_{}'.format(
              patient_id.decode('utf-8'), threshold), simple_value=f1_score)
          summary_writer.add_summary(summary, global_step=global_step)

          f2_score = (5 * precision * recall / (4 * precision + recall)) if (
            (4 * precision + recall)) > 0 else 0
          patient_f2_score[threshold].append(f2_score)
          summary = tf.Summary()
          summary.value.add(
            tag='patient_metrics/{}/f2_score_at_{}'.format(
              patient_id.decode('utf-8'), threshold), simple_value=f2_score)
          summary_writer.add_summary(summary, global_step=global_step)

    for threshold in self.tp_thresholds:
      threshold = int(np.round(threshold * 100))
      population_tnr = np.mean(patient_healthy_tnr[threshold])
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/healthy_tnr_at_{}'.format(threshold),
        simple_value=population_tnr)
      summary_writer.add_summary(summary, global_step=global_step)

      population_tnr = np.mean(patient_cancer_tnr[threshold])
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/cancer_tnr_at_{}'.format(threshold),
        simple_value=population_tnr)
      summary_writer.add_summary(summary, global_step=global_step)

      population_recall = np.mean(patient_recall[threshold])
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/recall_at_{}'.format(threshold),
        simple_value=population_recall)
      summary_writer.add_summary(summary, global_step=global_step)

      population_precision = np.mean(patient_precision[threshold])
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/precision_at_{}'.format(threshold),
        simple_value=population_precision)
      summary_writer.add_summary(summary, global_step=global_step)

      population_region_recall = np.mean(patient_region_recall[threshold])
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/region_recall_at_{}'.format(threshold),
        simple_value=population_region_recall)
      summary_writer.add_summary(summary, global_step=global_step)

      population_f1_score = np.mean(patient_f1_score[threshold])
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/f1_score_at_{}'.format(threshold),
        simple_value=population_f1_score)
      summary_writer.add_summary(summary, global_step=global_step)

      population_f2_score = np.mean(patient_f2_score[threshold])
      summary = tf.Summary()
      summary.value.add(
        tag='metrics/patient_adjusted/f2_score_at_{}'.format(threshold),
        simple_value=population_f2_score)
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
