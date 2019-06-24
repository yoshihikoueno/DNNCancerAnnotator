import os
import logging

import tensorflow as tf
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import image_utils


class Exam():
  def __init__(self, exam_id, statistics, num_lesions, froc_region_cm_values,
               num_slices, calc_froc):
    self.exam_id = exam_id
    self.statistics = statistics
    self.num_lesions = num_lesions
    self.num_slices = num_slices
    assert(self.num_slices > 0)
    self.froc_region_cm_values = froc_region_cm_values
    self.calc_froc = calc_froc

  def add_sample(self, statistics, num_lesions, froc_region_cm_values):
    for k, v in statistics.items():
      assert(k in self.statistics)
      self.statistics[k] += v

    self.num_lesions += num_lesions
    self.num_slices += 1

    if self.calc_froc:
      for k, threshold_values in froc_region_cm_values.items():
        assert(k in self.froc_region_cm_values)
        for i, v in enumerate(threshold_values):
          self.froc_region_cm_values[k][i] += v

  def get_normalized_num_lesions(self):
    return float(self.num_lesions) / float(self.num_slices)

  def get_normalized_froc_region_cm_values(self):
    normalized_froc_values = dict()
    for k, v in self.froc_region_cm_values.items():
      normalized_froc_values[k] = []
      for e in v:
        normalized_froc_values.append(float(e) / self.num_slices)

    return normalized_froc_values


class Patient():
  def __init__(self, patient_id, calc_froc):
    self.patient_id = patient_id
    self.calc_froc = calc_froc
    self.exams = dict()

  def set_sample(self, exam_id, statistics, num_lesions,
                 froc_region_cm_values):
    if exam_id not in self.exams:
      self.exams[exam_id] = Exam(
        exam_id=exam_id, statistics=statistics, num_lesions=num_lesions,
        froc_region_cm_values=froc_region_cm_values, num_slices=1,
        calc_froc=self.calc_froc)
    else:
      self.exams[exam_id].add_sample(
        statistics=statistics, num_lesions=num_lesions,
        froc_region_cm_values=froc_region_cm_values)

  def set_exam(self, exam_id, statistics, num_lesions, froc_region_cm_values,
               num_slices):
    assert(exam_id not in self.exams)

    self.exams[exam_id] = Exam(
      exam_id=exam_id, statistics=statistics, num_lesions=num_lesions,
      froc_region_cm_values=froc_region_cm_values, num_slices=num_slices,
      calc_froc=self.calc_froc)


class PatientMetricHandler():
  def __init__(self, eval_3d_as_2d, calc_froc, result_folder, eval_dir, is_3d,
               lesion_slice_ratio):
    self.eval_3d_as_2d = eval_3d_as_2d
    self.calc_froc = calc_froc
    self.patients = dict()
    self.result_folder = result_folder
    self.eval_dir = eval_dir
    self.is_3d = is_3d
    self.lesion_slice_ratio = lesion_slice_ratio

  def set_sample(self, patient_id, exam_id, statistics, num_lesions,
                 froc_region_cm_values):
    if (patient_id not in self.patients):
      self.patients[patient_id] = Patient(patient_id, calc_froc=self.calc_froc)

    self.patients[patient_id].set_sample(
      exam_id=exam_id, statistics=statistics, num_lesions=num_lesions,
      froc_region_cm_values=froc_region_cm_values)

  def set_exam(self, patient_id, exam_id, statistics, num_lesions,
               froc_region_cm_values, num_slices):
    if (patient_id not in self.patients):
      self.patients[patient_id] = Patient(patient_id, calc_froc=self.calc_froc)

    if self.is_3d and self.eval_3d_as_2d:
      num_lesions = np.sum(num_lesions)
      for k, v in statistics.items():
        statistics[k] = np.sum(v)

      if self.calc_froc:
        for k, v in froc_region_cm_values.items():
          froc_region_cm_values[k] = np.sum(v, axis=0)

    self.patients[patient_id].set_exam(
      exam_id=exam_id, statistics=statistics,
      num_lesions=num_lesions,
      froc_region_cm_values=froc_region_cm_values, num_slices=num_slices)
    logging.info("Patient {} exam {} finished.".format(patient_id, exam_id))

  def evaluate(self, global_step):
    logging.info("Starting final evaluation.")
    region_tp = 0.0
    region_fn = 0.0
    region_fp = 0.0
    tp = 0.0
    fn = 0.0
    fp = 0.0
    froc_total_cm_values = dict()
    num_total_lesions = 0

    for patient_id, patient in self.patients.items():
      patient_region_tp = 0.0
      patient_region_fn = 0.0
      patient_region_fp = 0.0
      patient_tp = 0.0
      patient_fn = 0.0
      patient_fp = 0.0
      froc_patient_cm_values = dict()
      num_patient_lesions = 0.0
      for exam_id, exam in patient.exams.items():
        patient_tp += float(exam.statistics['tp']) / exam.num_slices
        patient_fn += float(exam.statistics['fn']) / exam.num_slices
        patient_fp += float(exam.statistics['fp']) / exam.num_slices
        patient_region_tp += float(
          exam.statistics['region_tp']) / exam.num_slices
        patient_region_fn += float(
          exam.statistics['region_fn']) / exam.num_slices
        patient_region_fp += float(
          exam.statistics['region_fp']) / exam.num_slices
        num_patient_lesions += float(exam.num_lesions) / exam.num_slices
        if self.calc_froc:
          for k, threshold_values in exam.froc_region_cm_values.items():
            if k not in froc_patient_cm_values:
              froc_patient_cm_values[k] = [0.0] * len(threshold_values)
            for i, v in enumerate(threshold_values):
              froc_patient_cm_values[k][i] += float(v) / exam.num_slices

      assert(len(patient.exams.keys()) > 0)
      region_tp += patient_region_tp / len(patient.exams.keys())
      region_fn += patient_region_fn / len(patient.exams.keys())
      region_fp += patient_region_fp / len(patient.exams.keys())
      tp += patient_tp / len(patient.exams.keys())
      fn += patient_fn / len(patient.exams.keys())
      fp += patient_fp / len(patient.exams.keys())
      num_total_lesions += num_patient_lesions / len(patient.exams.keys())
      if self.calc_froc:
        for k, threshold_values in froc_patient_cm_values.items():
          if k not in froc_total_cm_values:
            froc_total_cm_values[k] = [0.0] * len(threshold_values)
          for i, v in enumerate(threshold_values):
            froc_total_cm_values[k][i] += float(v) / len(patient.exams.keys())

    adjusted_fp = 2 * fp * min(self.lesion_slice_ratio, 0.5)
    adjusted_region_fp = 2 * region_fp * min(self.lesion_slice_ratio, 0.5)

    summary_writer = tf.summary.FileWriterCache.get(
      os.path.join(self.result_folder, self.eval_dir))

    recall = tp / (tp + fn) if (tp + fn > 0) else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/recall', simple_value=recall)
    summary_writer.add_summary(summary, global_step=global_step)

    precision = tp / (tp + fp) if (tp + fp > 0) else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/precision',
      simple_value=precision)
    summary_writer.add_summary(summary, global_step=global_step)

    f1_score = (2 * precision * recall / (precision + recall)) if (
        precision + recall) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/f1_score',
      simple_value=f1_score)
    summary_writer.add_summary(summary, global_step=global_step)

    f2_score = (5 * precision * recall / (4 * precision + recall)) if (
      (4 * precision + recall)) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/f2_score',
      simple_value=f2_score)
    summary_writer.add_summary(summary, global_step=global_step)

    region_recall = region_tp / (region_tp + region_fn) if (
      region_tp + region_fn > 0) else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/region_recall',
      simple_value=region_recall)
    summary_writer.add_summary(summary, global_step=global_step)

    region_precision = region_tp / (region_tp + region_fp) if (
      region_tp + region_fp > 0) else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/region_precision',
      simple_value=region_precision)
    summary_writer.add_summary(summary, global_step=global_step)

    region_f1_score = (2 * region_precision * region_recall / (
      region_precision + region_recall)) if (
        region_precision + region_recall) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/region_f1_score',
      simple_value=region_f1_score)
    summary_writer.add_summary(summary, global_step=global_step)

    region_f2_score = (5 * region_precision * region_recall / (
      4 * region_precision + region_recall)) if (
        (4 * region_precision + region_recall)) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/region_f2_score',
      simple_value=region_f2_score)
    summary_writer.add_summary(summary, global_step=global_step)

    # Adjusted for lesion ratio metrics
    adjusted_precision = tp / (tp + adjusted_fp) if (
      tp + adjusted_fp > 0) else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/adjusted_precision',
      simple_value=adjusted_precision)
    summary_writer.add_summary(summary, global_step=global_step)

    adjusted_f1_score = (2 * adjusted_precision * recall / (
      adjusted_precision + recall)) if (
        adjusted_precision + recall) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/adjusted_f1_score',
      simple_value=adjusted_f1_score)
    summary_writer.add_summary(summary, global_step=global_step)

    adjusted_f2_score = (5 * adjusted_precision * recall / (
      4 * adjusted_precision + recall)) if (
      (4 * adjusted_precision + recall)) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/adjusted_f2_score',
      simple_value=adjusted_f2_score)
    summary_writer.add_summary(summary, global_step=global_step)

    adjusted_region_precision = region_tp / (
      region_tp + adjusted_region_fp) if (
        region_tp + adjusted_region_fp > 0) else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/adjusted_region_precision',
      simple_value=adjusted_region_precision)
    summary_writer.add_summary(summary, global_step=global_step)

    adjusted_region_f1_score = (
      2 * adjusted_region_precision * region_recall / (
        adjusted_region_precision + region_recall)) if (
          adjusted_region_precision + region_recall) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/adjusted_region_f1_score',
      simple_value=adjusted_region_f1_score)
    summary_writer.add_summary(summary, global_step=global_step)

    adjusted_region_f2_score = (
      5 * adjusted_region_precision * region_recall / (
        4 * adjusted_region_precision + region_recall)) if (
          (4 * adjusted_region_precision + region_recall)) > 0 else 0
    summary = tf.Summary()
    summary.value.add(
      tag='metrics/adjusted_region_f2_score',
      simple_value=adjusted_region_f2_score)
    summary_writer.add_summary(summary, global_step=global_step)

    if self.calc_froc and num_total_lesions > 0:
      # Plot FROC Curve
      # FP / Num Lesions
      x = []
      # TP / Num Lesions
      y = []

      region_fps = froc_total_cm_values['region_fp']
      region_tps = froc_total_cm_values['region_tp']
      for i in range(len(region_fps)):
        # Average number of FP per patient
        x.append(region_fps[i] / float(len(
          self.patients.keys())))
        y.append(region_tps[i] / float(num_total_lesions))

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
