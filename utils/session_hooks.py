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
      file_name = file_name_res[batch_index]

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


class PatientMetricHook:
  def __init__(self):
    pass

  def before_run(self, run_context):
    pass

  def after_run(self, run_context, run_values):
    pass
