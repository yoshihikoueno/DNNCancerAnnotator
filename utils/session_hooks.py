import logging

import tensorflow as tf
import numpy as np

from utils import standard_fields
from utils import image_utils


class VisualizationHook(tf.train.SessionRunHook):
  def __init__(self, result_folder, num_visualizations, num_total_examples,
               feature_dict, predicted_masks):
    if num_visualizations is None or num_visualizations == -1:
      self.num_visualizations = num_total_examples
    else:
      self.num_visualizations = min(num_visualizations, num_total_examples)

    self.visualization_indices = np.random.choice(np.arange(
      num_visualizations), num_visualizations)
    self.first_epoch_index = 0
    self.num_total_examples = num_total_examples
    # The file names will be filled in the first epoch. This is necessary so
    # that in consecutive epochs, even shuffled datasets have the same
    # visualized images
    self.visualization_file_names = []
    self.result_folder = result_folder
    self.feature_dict = feature_dict
    background, foreground = tf.split(tf.sigmoid(predicted_masks), 2, axis=3)

    target_size = background.get_shape().as_list()[1:3]

    image_decoded = image_utils.central_crop(
      feature_dict[standard_fields.InputDataFields.image_decoded], target_size)

    gt_masks = image_utils.central_crop(
      feature_dict[standard_fields.InputDataFields.annotation_decoded],
      target_size)

    background = tf.image.grayscale_to_rgb(background) * 255
    foreground = tf.image.grayscale_to_rgb(foreground) * 255
    image_decoded = tf.image.grayscale_to_rgb(image_decoded)

    self.combined_image = tf.concat([image_decoded, gt_masks, background,
                                     foreground], axis=2)

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=[
      self.feature_dict, self.combined_image, tf.train.get_global_step()])

  def after_run(self, run_context, run_values):
    feature_dict_res = run_values.results[0]
    combined_image_res = run_values.results[1]
    global_step = run_values.results[2]

    summary_writer = tf.summary.FileWriterCache.get(self.result_folder)

    for batch_index in range(len(combined_image_res)):
      file_name = feature_dict_res[
            standard_fields.InputDataFields.image_file][batch_index]

      if self.first_epoch_index < self.num_total_examples:
        if self.first_epoch_index in self.visualization_indices:
          self.visualization_file_names.append(file_name)
          self.first_epoch_index += 1
        else:
          self.first_epoch_index += 1
          continue
      else:
        if file_name not in self.visualization_file_names:
          continue

      summary = tf.Summary(value=[
          tf.Summary.Value(
            tag=file_name,
            image=tf.Summary.Image(
              encoded_image_string=image_utils.encode_image_array_as_png_str(
                combined_image_res[batch_index])))])
      summary_writer.add_summary(summary, global_step)

      logging.info('Detection visualizations written to summary with tag %s.',
                   file_name)
