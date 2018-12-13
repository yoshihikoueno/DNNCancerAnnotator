import functools
import logging

import numpy as np
import tensorflow as tf

from models import unet
from utils import standard_fields
from utils import preprocessor
from utils import session_hooks
from utils import metric_utils
from utils import image_utils
from utils import util_ops
from builders import optimizer_builder


def _loss(labels, logits, loss_name, pos_weight):
  # Each entry in labels must be an index in [0, num_classes)
  assert(len(labels.get_shape()) == 1)

  if loss_name == 'sigmoid':
    assert(logits.get_shape().as_list()[1] == 1)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
      tf.to_float(labels), tf.squeeze(logits, axis=1), pos_weight))
  elif loss_name == 'softmax':
    # Logits should be of shape [batch_size, num_classes]
    assert(len(logits.get_shape()) == 2)
    assert(pos_weight is None and 'Loss weight not implemented for softmax.')
    return tf.losses.sparse_softmax_cross_entropy(labels, logits)
  else:
    assert(False and 'Loss name "{}" not recognized.'.format(loss_name))


def _general_model_fn(features, pipeline_config, result_folder, dataset_info,
                      dataset_split_name, model_constructor, mode, num_gpu,
                      visualization_file_names):
  add_background_class = pipeline_config.train_config.loss.name == 'softmax'
  net = model_constructor(pipeline_config, is_training=mode
                          == tf.estimator.ModeKeys.TRAIN,
                          add_background_class=add_background_class)

  image_batch = features[standard_fields.InputDataFields.image_decoded]

  annotation_mask_batch = features[
    standard_fields.InputDataFields.annotation_mask]

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Preprocess / Augment images
    image_batch, annotation_mask_batch = preprocessor.apply(
      pipeline_config.train_config.data_augmentation_options,
      images=image_batch, gt_masks=annotation_mask_batch,
      batch_size=pipeline_config.train_config.batch_size)

  network_output = net.build_network(image_batch)

  annotation_mask_batch = tf.cast(tf.clip_by_value(image_utils.central_crop(
    annotation_mask_batch, desired_size=network_output.get_shape().as_list()[
      1:3]), 0, 1), dtype=tf.int64)
  assert(len(annotation_mask_batch.get_shape()) == 4)

  patient_ratio = dataset_info[
    standard_fields.PickledDatasetInfo.patient_ratio]
  cancer_pixels = tf.reduce_sum(tf.to_float(annotation_mask_batch))
  healthy_pixels = tf.to_float(tf.size(annotation_mask_batch)) - cancer_pixels

  batch_pixel_ratio = tf.div(healthy_pixels, cancer_pixels + 1.0)

  loss = _loss(tf.reshape(annotation_mask_batch, [-1]),
                 tf.reshape(network_output, [-1, net.num_classes]),
                 loss_name=pipeline_config.train_config.loss.name,
                 pos_weight=batch_pixel_ratio * patient_ratio)
  loss = tf.identity(loss, name='ModelLoss')
  tf.summary.scalar(loss.op.name, loss, family='Loss')

  total_loss = tf.identity(loss, name='TotalLoss')
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
  train_op = tf.identity(total_loss, name='train_op')
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

  logging.info("Total number of trainable parameters: {}".format(np.sum([
    np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Training Hooks are not working with MirroredStrategy. Fixed in 1.13
    #print_hook = session_hooks.PrintHook(
    #  file_name=features[standard_fields.InputDataFields.image_file],
    #  batch_pixel_ratio=batch_pixel_ratio)
    return tf.estimator.EstimatorSpec(mode,
                                      loss=total_loss, train_op=train_op,
                                      scaffold=scaffold)
  elif mode == tf.estimator.ModeKeys.EVAL:
    if pipeline_config.train_config.loss.name == 'sigmoid':
      scaled_network_output = tf.nn.sigmoid(network_output)[:, :, :, 0]
    elif pipeline_config.train_config.loss.name == 'softmax':
      assert(network_output.get_shape().as_list()[-1] == 2)
      scaled_network_output = tf.nn.softmax(network_output)[:, :, :, 1]

      # Metrics
    metric_dict, region_statistics_dict = metric_utils.get_metrics(
      scaled_network_output, annotation_mask_batch,
      tp_thresholds=np.array(pipeline_config.metrics_tp_thresholds,
                             dtype=np.float32),
      parallel_iterations=min(pipeline_config.eval_config.batch_size,
                              util_ops.get_cpu_count()))

    vis_hook = session_hooks.VisualizationHook(
      result_folder=result_folder,
      visualization_file_names=visualization_file_names,
      file_name=features[standard_fields.InputDataFields.image_file],
      image_decoded=image_batch,
      annotation_decoded=features[
        standard_fields.InputDataFields.annotation_decoded],
      annotation_mask=annotation_mask_batch,
      predicted_mask=scaled_network_output)
    patient_metric_hook = session_hooks.PatientMetricHook(
      region_statistics_dict=region_statistics_dict,
      patient_id=features[standard_fields.InputDataFields.patient_id],
      result_folder=result_folder)

    return tf.estimator.EstimatorSpec(
      mode, loss=total_loss, evaluation_hooks=[vis_hook, patient_metric_hook],
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

  np.random.shuffle(visualization_file_names)

  visualization_file_names = visualization_file_names[:num_visualizations]

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
