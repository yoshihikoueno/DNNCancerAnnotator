import logging
import os

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from utils import image_utils


class VisualizationHook(tf.train.SessionRunHook):
  def __init__(self, result_folder, visualization_file_names,
               file_name, image_decoded, annotation_decoded,
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

    predicted_mask_overlay = tf.clip_by_value(
      image_decoded * 0.5 + predicted_mask, 0, 255)

    if annotation_decoded is None:
      # Predict Mode
      self.combined_image = tf.concat([
        image_decoded, predicted_mask_overlay, predicted_mask], axis=2)
    else:
      annotation_decoded = image_utils.central_crop(
        annotation_decoded, target_size)
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
      self.eval_dir)

    for batch_index in range(len(combined_image_res)):
      file_name = file_name_res[batch_index].decode('utf-8')

      # In prediction mode we want to visualize in any case
      if (self.visualization_file_names is None
          or file_name in self.visualization_file_names):
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
               eval_dir, num_lesions, froc_region_cm_values, froc_thresholds):
    self.statistics_dict = statistics_dict
    self.patient_id = patient_id
    self.result_folder = result_folder
    self.eval_dir = eval_dir
    self.num_lesions = num_lesions
    self.froc_region_cm_values = froc_region_cm_values
    self.froc_thresholds = froc_thresholds

    self.patient_statistics = dict()
    self.num_total_lesions = 0
    # Threshold to cm values
    self.froc_cm_values_total = dict()

  def before_run(self, run_context):
    fetch_list = [self.statistics_dict,
                  self.patient_id,
                  self.num_lesions]
    if self.froc_region_cm_values is not None:
      fetch_list.append(self.froc_region_cm_values)

    return tf.train.SessionRunArgs(fetches=fetch_list)

  def after_run(self, run_context, run_values):
    statistics_dict_res = run_values.results[0]
    patient_id_res = run_values.results[1]
    num_lesions_res = run_values.results[2]
    if self.froc_region_cm_values is None:
      froc_region_cm_values_res = None
    else:
      froc_region_cm_values_res = run_values.results[3]

    self.num_total_lesions += num_lesions_res

    for patient_id in patient_id_res:
      if patient_id not in self.patient_statistics:
        self.patient_statistics[patient_id] = dict()
        self.patient_statistics[patient_id]['num_slices'] = 0

      self.patient_statistics[patient_id]['num_slices'] += 1

    for confusion_key, batch_values in statistics_dict_res.items():
      for i, val in enumerate(batch_values):
        if confusion_key not in self.patient_statistics[patient_id_res[i]]:
          self.patient_statistics[patient_id_res[i]][confusion_key] = 0

        self.patient_statistics[patient_id_res[i]][confusion_key] += val

    if froc_region_cm_values_res:
      for i, threshold in enumerate(self.froc_thresholds):
        if threshold not in self.froc_cm_values_total:
          self.froc_cm_values_total[threshold] = dict()
          for k in froc_region_cm_values_res.keys():
            self.froc_cm_values_total[threshold][k] = 0

        for k, v in froc_region_cm_values_res.items():
          self.froc_cm_values_total[threshold][k] += v[i]

  def end(self, session):
    # Collect all normalized confusion values
    region_tp = 0
    region_fn = 0
    region_fp = 0
    tp = 0
    fp = 0
    fn = 0

    summary_writer = tf.summary.FileWriterCache.get(
      os.path.join(self.result_folder, self.eval_dir))
    global_step = tf.train.get_global_step().eval(session=session)

    for patient_id, statistics in self.patient_statistics.items():
      num_slices = statistics['num_slices']
      assert(num_slices > 0)

      region_tp += (statistics['region_tp']
                               / float(num_slices))
      region_fn += (statistics['region_fn']
                               / float(num_slices))
      region_fp += (statistics['region_fp']
                               / float(num_slices))
      tp += (statistics['tp'] / float(num_slices))
      fp += (statistics['fp'] / float(num_slices))
      fn += (statistics['fn'] / float(num_slices))

    recall = tp / (tp + fn) if (tp + fn > 0) else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/patient_adjusted/population_recall', simple_value=recall)
    summary_writer.add_summary(summary, global_step=global_step)

    precision = tp / (tp + fp) if (tp + fp > 0) else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/patient_adjusted/population_precision',
      simple_value=precision)
    summary_writer.add_summary(summary, global_step=global_step)

    f1_score = (2 * precision * recall / (precision + recall)) if (
        precision + recall) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/patient_adjusted/population_f1_score',
      simple_value=f1_score)
    summary_writer.add_summary(summary, global_step=global_step)

    f2_score = (5 * precision * recall / (4 * precision + recall)) if (
      (4 * precision + recall)) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/patient_adjusted/population_f2_score',
      simple_value=f2_score)
    summary_writer.add_summary(summary, global_step=global_step)

    region_recall = region_tp / (region_tp + region_fn) if (
      region_tp + region_fn > 0) else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/patient_adjusted/population_region_recall',
      simple_value=region_recall)
    summary_writer.add_summary(summary, global_step=global_step)

    region_precision = region_tp / (region_tp + region_fp) if (
      region_tp + region_fp > 0) else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/patient_adjusted/population_region_precision',
      simple_value=region_precision)
    summary_writer.add_summary(summary, global_step=global_step)

    region_f1_score = (2 * region_precision * region_recall / (
      region_precision + region_recall)) if (
        region_precision + region_recall) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/patient_adjusted/population_region_f1_score',
      simple_value=region_f1_score)
    summary_writer.add_summary(summary, global_step=global_step)

    region_f2_score = (5 * region_precision * region_recall / (
      4 * region_precision + region_recall)) if (
        (4 * region_precision + region_recall)) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/patient_adjusted/population_region_f2_score',
      simple_value=region_f2_score)
    summary_writer.add_summary(summary, global_step=global_step)

    if self.froc_region_cm_values and self.num_total_lesions > 0:
      # Plot FROC Curve
      # FP / Num Lesions
      x = []
      # TP / Num Lesions
      y = []

      for threshold, cm_values in self.froc_cm_values_total.items():
        # Average number of FP per patient
        x.append(cm_values['region_fp'] / float(len(
          self.patient_statistics.keys())))
        y.append(cm_values['region_tp'] / float(self.num_total_lesions))

      fig = Figure()
      canvas = FigureCanvas(fig)
      ax = fig.gca(ylim=(0, 1),
                   yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                           1.0],
                   xlabel='Average number of FP', ylabel='True Positive Rate')
      ax.grid(True)

      ax.plot(x, y)

      canvas.draw()

      width, height = fig.get_size_inches() * fig.get_dpi()

      plot_img = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape(
        (int(height), int(width), 3))

      logging.info("Visualizing FROC Curve.")
      summary = tf.Summary(value=[
        tf.Summary.Value(
          tag='FROC_Curve',
          image=tf.Summary.Image(
            encoded_image_string=image_utils.encode_image_array_as_png_str(
              plot_img)))])

      summary_writer.add_summary(summary, global_step=global_step)

    else:
      logging.warn('Number of total lesions is 0. No FROC Curve plotted.')


class PrintHook(tf.train.SessionRunHook):
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=self.kwargs)

  def after_run(self, run_context, run_values):
    kwargs_res = run_values.results

    for key, value in kwargs_res.items():
      logging.info("{}: {}".format(key, value))
