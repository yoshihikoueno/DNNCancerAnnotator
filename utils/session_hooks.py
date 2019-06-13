import logging
import os

import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from utils import image_utils
from utils import metric_utils
from utils import util_ops
from metrics import patient_metric_handler


class Eval3DHook(tf.train.SessionRunHook):
  def __init__(
      self, groundtruth, prediction, slice_ids, patient_id, exam_id,
      eval_3d_as_2d, patient_exam_id_to_num_slices, calc_froc, target_size,
      result_folder, eval_dir, lesion_slice_ratio):
    self.groundtruth = groundtruth
    self.prediction = prediction
    self.slice_ids = slice_ids
    self.patient_id = patient_id
    self.exam_id = exam_id
    self.eval_3d_as_2d = eval_3d_as_2d
    self.patient_exam_id_to_num_slices = patient_exam_id_to_num_slices
    self.calc_froc = calc_froc
    self.target_size = target_size

    self.current_patient_id = None
    self.current_exam_id = None
    self.full_groundtruth = None
    self.full_prediction = None
    self.first_slice_id = None
    self.last_slice_id = None

    self.groundtruth_op = tf.placeholder(
      shape=[None, self.target_size[0],
             self.target_size[1]], dtype=tf.float32)
    self.prediction_op = tf.placeholder(
      shape=[None, self.target_size[0],
             self.target_size[1]], dtype=tf.float32)

    self.patient_metric_handler = (
      patient_metric_handler.PatientMetricHandler(
        eval_3d_as_2d=self.eval_3d_as_2d, calc_froc=self.calc_froc,
        result_folder=result_folder, eval_dir=eval_dir, is_3d=True,
        lesion_slice_ratio=lesion_slice_ratio))

    self.eval_ops = list(self._make_eval_op())
    self.eval_ops.append(self.groundtruth_op)

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=[
      self.groundtruth, self.prediction, self.slice_ids, self.patient_id,
      self.exam_id])

  def after_run(self, run_context, run_values):
    groundtruth_res = run_values.results[0]
    prediction_res = run_values.results[1]
    slice_ids_res = run_values.results[2]
    patient_id_res = run_values.results[3].decode('utf-8')
    exam_id_res = run_values.results[4].decode('utf-8')

    for i, slice_id in enumerate(slice_ids_res):
      if self.current_patient_id is None:
        if slice_id == -1:
          continue
        self.current_patient_id = patient_id_res
        self.current_exam_id = exam_id_res
        self.first_slice_id = slice_id
        self.full_groundtruth = [None] * self.patient_exam_id_to_num_slices[
          self.current_patient_id][self.current_exam_id]
        self.full_prediction = [None] * self.patient_exam_id_to_num_slices[
          self.current_patient_id][self.current_exam_id]
      assert(slice_id is not None)

      assert(patient_id_res == self.current_patient_id)
      assert(exam_id_res == self.current_exam_id)

      assert(self.full_groundtruth[slice_id - self.first_slice_id] is None)
      assert(self.full_prediction[slice_id - self.first_slice_id] is None)

      if self.last_slice_id is not None:
        assert(self.last_slice_id + 1 == slice_id)

      self.last_slice_id = slice_id

      self.full_groundtruth[slice_id - self.first_slice_id] = groundtruth_res[
        i]
      self.full_prediction[slice_id - self.first_slice_id] = prediction_res[i]

      if (slice_id - self.first_slice_id + 1 == len(self.full_groundtruth)):
        # We have all slices
        self._evaluate_current_patient_exam(run_context.session)

        self.current_patient_id = None
        self.current_exam_id = None
        self.last_slice_id = None

  def end(self, session):
    self.patient_metric_handler.evaluate(
      tf.train.get_global_step().eval(session=session))

  def _make_eval_op(self):
    prediction_groundtruth_stack = tf.stack(
      [self.prediction_op, tf.cast(self.groundtruth_op, tf.float32)],
      axis=1 if self.eval_3d_as_2d else 0)

    if self.eval_3d_as_2d:
      return tf.map_fn(lambda e: metric_utils.get_metrics(
        e,
        parallel_iterations=util_ops.get_cpu_count(),
        calc_froc=self.calc_froc, is_3d=False),
                       elems=prediction_groundtruth_stack, dtype=(
                         {}, {'tp': tf.int64, 'fp': tf.int64, 'fn': tf.int64,
                              'region_tp': tf.int64, 'region_fp': tf.int64,
                              'region_fn': tf.int64}, tf.int64, {
                                'region_tp': tf.int64, 'region_fp': tf.int64,
                                'region_fn': tf.int64},
                         tf.float32))
    else:
      return metric_utils.get_metrics(
        prediction_groundtruth_stack,
        parallel_iterations=util_ops.get_cpu_count(),
        calc_froc=self.calc_froc, is_3d=True)

  def _evaluate_current_patient_exam(self, sess):
    # Make sure all elements are not None
    for g, p in zip(self.full_groundtruth, self.full_prediction):
      assert(g is not None)
      assert(p is not None)

    (metric_dict, statistics_dict, num_lesions, froc_region_cm_values,
       froc_thresholds, groundtruth) = sess.run(
         self.eval_ops, feed_dict={self.groundtruth_op: self.full_groundtruth,
                                    self.prediction_op: self.full_prediction})

    num_slices = groundtruth.shape[0]

    self.patient_metric_handler.set_exam(
      patient_id=self.current_patient_id, exam_id=self.current_exam_id,
      statistics=statistics_dict, num_lesions=num_lesions,
      froc_region_cm_values=froc_region_cm_values, num_slices=num_slices)


class VisualizationHook(tf.train.SessionRunHook):
  def __init__(self, result_folder, visualization_file_names,
               file_name, image_decoded, annotation_decoded,
               predicted_mask, eval_dir, is_3d):
    self.visualization_file_names = visualization_file_names
    self.result_folder = result_folder
    self.file_name = file_name
    self.eval_dir = eval_dir
    self.is_3d = is_3d
    if is_3d:
      assert(len(predicted_mask.get_shape()) == 3)
    else:
      assert(len(predicted_mask.get_shape()) == 2)
      self.file_name = tf.expand_dims(self.file_name, axis=0)
      self.image_decoded = tf.expand_dims(self.image_decoded, axis=0)
      self.annotation_decoded = tf.expand_dims(self.annotation_decoded, axis=0)
      self.predicted_mask = tf.expand_dims(self.predicted_mask, axis=0)
      self.is_3d

    target_size = predicted_mask.get_shape().as_list()[1:]

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

    print(file_name_res)
    for batch_index in range(len(combined_image_res)):
      file_name = file_name_res[batch_index].decode('utf-8')

      if file_name == '':
        continue

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
               eval_dir, num_lesions, froc_region_cm_values, froc_thresholds,
               calc_froc, lesion_slice_ratio):
    self.statistics_dict = statistics_dict
    self.patient_id = patient_id
    self.result_folder = result_folder
    self.eval_dir = eval_dir
    self.num_lesions = num_lesions
    self.calc_froc = calc_froc
    self.froc_region_cm_values = froc_region_cm_values
    self.froc_thresholds = froc_thresholds

    self.patient_metric_handler = patient_metric_handler.PatientMetricHandler(
      eval_3d_as_2d=False, calc_froc=calc_froc, result_folder=result_folder,
      eval_dir=eval_dir, is_3d=False, lesion_slice_ratio=lesion_slice_ratio)

  def before_run(self, run_context):
    fetch_list = [self.statistics_dict,
                  self.patient_id,
                  self.exam_id,
                  self.num_lesions]
    if self.calc_froc:
      fetch_list.append(self.froc_region_cm_values)

    return tf.train.SessionRunArgs(fetches=fetch_list)

  def after_run(self, run_context, run_values):
    statistics_dict_res = run_values.results[0]
    patient_id_res = run_values.results[1]
    exam_id_res = run_values.results[2]
    num_lesions_res = run_values.results[3]

    if self.calc_froc:
      froc_region_cm_values_res = None
    else:
      froc_region_cm_values_res = run_values.results[4]

    self.patient_metric_handler.set_sample(
      patient_id=patient_id_res, exam_id=exam_id_res,
      statistics=statistics_dict_res, num_lesions=num_lesions_res,
      froc_region_cm_values=froc_region_cm_values_res)

  def end(self, session):
    self.patient_metric_handler.evaluate(
      tf.train.get_global_step().eval(session=session))


class PrintHook(tf.train.SessionRunHook):
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=self.kwargs)

  def after_run(self, run_context, run_values):
    kwargs_res = run_values.results

    for key, value in kwargs_res.items():
      logging.info("{}: {}".format(key, value))
