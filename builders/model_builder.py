import functools
import logging

import numpy as np
import tensorflow as tf

from models import unet
from utils import standard_fields
from utils import preprocessor
from utils import session_hooks
from utils import metrics
from utils import image_utils
from builders import optimizer_builder


def _general_model_fn(features, pipeline_config, result_folder, dataset_info,
                      dataset_split_name, model_constructor, mode, num_gpu,
                      visualization_file_names):
  net = model_constructor(pipeline_config, is_training=mode
                          == tf.estimator.ModeKeys.TRAIN)

  image_batch = features[standard_fields.InputDataFields.image_decoded]

  annotation_mask_batch = features[
    standard_fields.InputDataFields.annotation_mask]

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Preprocess / Augment images
    image_batch, annotation_mask_batch = preprocessor.apply(
      pipeline_config.train_config.data_augmentation_options,
      images=image_batch, gt_masks=annotation_mask_batch)

  network_output = net.build_network(image_batch)

  annotation_mask_batch = tf.clip_by_value(image_utils.central_crop(
      annotation_mask_batch,
      desired_size=network_output.get_shape().as_list()[1:3]), 0, 1)

  losses_dict = net.loss(network_output, annotation_mask_batch)

  loss = tf.add_n(list(losses_dict.values()), name='ModelLoss')
  tf.summary.scalar(loss.op.name, loss, family='Loss')
  total_loss = tf.identity(loss, name='Total_Loss')

  if mode == tf.estimator.ModeKeys.TRAIN:
    if pipeline_config.train_config.add_regularization_loss:
      regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
      if regularization_losses:
        regularization_loss = tf.add_n(regularization_losses,
                                       name='RegularizationLoss')
        total_loss = tf.add_n([loss, regularization_loss],
                              name='TotalLoss')
        tf.summary.scalar(regularization_loss.op.name, regularization_loss,
                          family='Loss')

  tf.summary.scalar(total_loss.op.name, total_loss, family='Loss')
  total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

  scaffold = None
  train_op = tf.identity(total_loss)
  if mode == tf.estimator.ModeKeys.TRAIN:
    if pipeline_config.train_config.optimizer.use_moving_average:
      # EMA's are currently not supported with tf's DistributionStrategy.
      # Reenable once they fixed the bugs
      logging.warn(
        'EMA is currently not supported with tf DistributionStrategy.')
      pipeline_config.train_config.optimizer.use_moving_average = False
      # The swapping saver will swap the trained variables with their moving
      # averages before saving, thus removing the need to care for moving
      # averages during evaluation
      # scaffold = tf.train.Scaffold(saver=optimizer.swapping_saver())

    optimizer, optimizer_summary_vars = optimizer_builder.build(
      pipeline_config.train_config.optimizer)
    for var in optimizer_summary_vars:
      tf.summary.scalar(var.op.name, var, family='LearningRate')

    grads_and_vars = optimizer.compute_gradients(total_loss)

    gradient_updates = optimizer.apply_gradients(
        grads_and_vars, global_step=tf.train.get_global_step())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append(gradient_updates)
    update_op = tf.group(*update_ops, name='update_barrier')
    with tf.control_dependencies([update_op]):
      train_op = tf.identity(total_loss)

  # Metrics
  metric_dict = metrics.get_metrics(
    network_output, annotation_mask_batch,
    thresholds=np.array(pipeline_config.eval_config.metric_thresholds,
                        dtype=np.float32))

  logging.info("Total number of trainable parameters: {}".format(np.sum([
    np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

  if mode == tf.estimator.ModeKeys.TRAIN:
    for k, v in metric_dict.items():
      tf.summary.scalar(k, v[1])
    return tf.estimator.EstimatorSpec(mode,
                                      loss=total_loss, train_op=train_op,
                                      scaffold=scaffold)
  elif mode == tf.estimator.ModeKeys.EVAL:
    batch_size = pipeline_config.eval_config.batch_size
    if num_gpu > 0:
      batch_size *= num_gpu

    hook = session_hooks.VisualizationHook(
      result_folder=result_folder,
      num_visualizations=pipeline_config.eval_config.num_images_to_visualize,
      num_total_examples=dataset_info[
        standard_fields.PickledDatasetInfo.split_to_size][dataset_split_name],
      batch_size=batch_size,
      file_name=features[standard_fields.InputDataFields.image_file],
      image_decoded=image_batch,
      annotation_decoded=features[
        standard_fields.InputDataFields.annotation_decoded],
      annotation_mask=annotation_mask_batch,
      predicted_masks=network_output)
    return tf.estimator.EstimatorSpec(
      mode, loss=total_loss, evaluation_hooks=[hook],
      eval_metric_ops=metric_dict)
  else:
    assert(False)


def get_model_fn(pipeline_config, result_folder, dataset_info,
                 dataset_split_name, num_gpu):
  visualization_file_names = dataset_info[
    standard_fields.PickledDatasetInfo.file_names][dataset_split_name]
  num_visualizations = pipeline_config.eval_config.num_images_to_visualize
  if num_visualizations is None or num_visualizations == -1:
    num_visualizations = len(visualization_file_names)
  else:
    num_visualizations = min(num_visualizations,
                             len(visualization_file_names))

  # Choose at random num_visualizations file names to visualize
  visualization_file_names = np.array(visualization_file_names)[
    np.random.choice(np.arange(len(visualization_file_names)),
                     num_visualizations)]

  model_name = pipeline_config.model.WhichOneof('model_type')
  if model_name == 'unet':
    return functools.partial(_general_model_fn,
                             pipeline_config=pipeline_config,
                             result_folder=result_folder,
                             dataset_info=dataset_info,
                             dataset_split_name=dataset_split_name,
                             model_constructor=unet.UNet,
                             num_gpu=num_gpu,
                             visualization_file_names=visualization_file_names)
  else:
    assert(False)
