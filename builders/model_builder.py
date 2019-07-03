import functools
import logging
import os

import numpy as np
import tensorflow as tf

from models import unet, unet_pix2pix, gan_discriminator
from utils import standard_fields
from utils import preprocessor
from utils import session_hooks
from utils import metric_utils
from utils import image_utils
from utils import util_ops
from builders import optimizer_builder


def _extract_patient_id(file_name):
  tokens = file_name.split('/')
  assert(tokens[0] == 'healthy_cases' or tokens[0] == 'cancer_cases')
  is_healthy = tokens[0] == 'healthy_cases'

  patient_id_prefix = 'h' if is_healthy else 'c'
  patient_id = patient_id_prefix + tokens[1]

  return patient_id


def _loss(labels, logits, loss_name, weight, is_pos_weight):
  # Each entry in labels must be an index in [0, num_classes)
  assert(len(labels.get_shape()) == 1)

  if loss_name == 'sigmoid':
    assert(logits.get_shape().as_list()[1] == 1)
    logits = tf.squeeze(logits, axis=1)
    if is_pos_weight:
      return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        labels=tf.cast(labels, dtype=tf.float32), logits=logits,
        pos_weight=weight))
    else:
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, dtype=tf.float32), logits=logits)
      weight_mask = tf.where(tf.equal(labels, 0), tf.fill(
        tf.shape(labels), weight), tf.fill(tf.shape(labels), 1.0))

      return tf.reduce_mean(tf.multiply(loss, weight_mask))
  elif loss_name == 'softmax':
    # Logits should be of shape [batch_size, num_classes]
    assert(len(logits.get_shape()) == 2)
    assert(weight is None and 'Loss weight not implemented for softmax.')
    return tf.losses.sparse_softmax_cross_entropy(labels, logits)
  else:
    assert(False and 'Loss name "{}" not recognized.'.format(loss_name))


def _gan_discriminator_model_fn(generated_data, conditioning,
                                model,
                                use_batch_norm, bn_momentum, bn_epsilon):
  print(generated_data)
  print(conditioning)
  return model.build_network(
    tf.concat([generated_data, conditioning], axis=-1), is_training=True,
    num_classes=2, use_batch_norm=use_batch_norm, bn_momentum=bn_momentum,
    bn_epsilon=bn_epsilon)


def _general_model_fn(features, mode, calc_froc, pipeline_config,
                      result_folder,
                      dataset_info, feature_extractor,
                      visualization_file_names, eval_dir,
                      as_gan_generator, eval_split_name):
  num_classes = pipeline_config.dataset.num_classes
  add_background_class = pipeline_config.train_config.loss.name == 'softmax'
  if add_background_class:
    assert(num_classes == 1)
    num_classes += 1

  if as_gan_generator:
    image_batch = features
  else:
    image_batch = features[standard_fields.InputDataFields.image_preprocessed]

  network_output = feature_extractor.build_network(
    image_batch, is_training=mode == tf.estimator.ModeKeys.TRAIN,
    num_classes=num_classes,
    use_norm=pipeline_config.model.use_norm,
    norm_config=pipeline_config.model)

  if as_gan_generator:
    return network_output

  if mode == tf.estimator.ModeKeys.PREDICT:
    annotation_mask_batch = None
  else:
    annotation_mask_batch = features[
      standard_fields.InputDataFields.annotation_mask]

  print(annotation_mask_batch)

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Record model variable summaries
    for var in tf.trainable_variables():
      tf.summary.histogram('ModelVars/' + var.op.name, var)

  network_output_shape = network_output.get_shape().as_list()
  if mode != tf.estimator.ModeKeys.PREDICT:
    if (network_output_shape[1:-1]
        != annotation_mask_batch.get_shape().as_list()[1:-1]):
      annotation_mask_batch = image_utils.central_crop(
        annotation_mask_batch,
        desired_size=network_output.get_shape().as_list()[1:-1])

    annotation_mask_batch = tf.cast(
      tf.clip_by_value(annotation_mask_batch, 0, 1), dtype=tf.int64)

    assert(annotation_mask_batch.get_shape().as_list()[:-1]
           == network_output.get_shape().as_list()[:-1])

  # We should not apply the loss to evaluation. This would just cause
  # our loss to be minimum for f2 score, but we also get the same
  # optimum if we just optimize for f1 score
  if (pipeline_config.train_config.loss.use_weighted
      and mode == tf.estimator.ModeKeys.TRAIN):
    cancer_pixels = tf.reduce_sum(tf.cast(annotation_mask_batch,
                                          dtype=tf.float32))
    healthy_pixels = tf.cast(tf.size(
      annotation_mask_batch), dtype=tf.float32) - cancer_pixels

    batch_pixel_ratio = tf.div(healthy_pixels, cancer_pixels + 1.0)

    loss_weight = (
      (batch_pixel_ratio
       + pipeline_config.train_config.loss.weight_constant_add)
      * pipeline_config.train_config.loss.weight_constant_multiply)
  else:
    loss_weight = tf.constant(1.0)

  if mode == tf.estimator.ModeKeys.PREDICT:
    loss = None
  else:
    loss = _loss(tf.reshape(annotation_mask_batch, [-1]),
                 tf.reshape(network_output, [-1, num_classes]),
                 loss_name=pipeline_config.train_config.loss.name,
                 weight=loss_weight,
                 is_pos_weight=pipeline_config.train_config.loss.pos_weight)
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
  update_ops = []
  if mode == tf.estimator.ModeKeys.TRAIN:
    if pipeline_config.train_config.optimizer.use_moving_average:
      # EMA's are currently not supported with tf's DistributionStrategy.
      # Reenable once they fixed the bugs
      logging.warn(
        'EMA is currently not supported with tf DistributionStrategy.')
      exit(1)
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

    update_ops.append(optimizer.apply_gradients(
      grads_and_vars, global_step=tf.train.get_global_step()))

  graph_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  update_ops.append(graph_update_ops)
  update_op = tf.group(*update_ops, name='update_barrier')
  with tf.control_dependencies([update_op]):
    if mode == tf.estimator.ModeKeys.PREDICT:
      train_op = None
    else:
      train_op = tf.identity(total_loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    logging.info("Total number of trainable parameters: {}".format(np.sum([
      np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

    # Training Hooks are not working with MirroredStrategy. Fixed in 1.13
    #print_hook = session_hooks.PrintHook(
    #  file_name=features[standard_fields.InputDataFields.image_file],
    #  batch_pixel_ratio=batch_pixel_ratio)
    return tf.estimator.EstimatorSpec(mode,
                                      loss=total_loss, train_op=train_op,
                                      scaffold=scaffold)
  elif mode == tf.estimator.ModeKeys.EVAL:
    image_decoded = features[standard_fields.InputDataFields.image_decoded]
    if pipeline_config.train_config.loss.name == 'sigmoid':
      if pipeline_config.dataset.tfrecords_type == 'input_3d':
        scaled_network_output = tf.nn.sigmoid(network_output)[:, :, :, :, 0]
      else:
        scaled_network_output = tf.nn.sigmoid(network_output)[:, :, :, 0]
    elif pipeline_config.train_config.loss.name == 'softmax':
      # We dont care about softmax for now
      assert(False)
      assert(network_output.get_shape().as_list()[-1] == 2)
      scaled_network_output = tf.nn.softmax(network_output)[:, :, :, 1]

    scaled_network_output = tf.squeeze(scaled_network_output, axis=0)
    annotation_mask = tf.squeeze(annotation_mask_batch, axis=0)
    image_decoded = tf.squeeze(image_decoded, axis=0)
    slice_ids = tf.squeeze(
      features[standard_fields.InputDataFields.slice_id], axis=0)
    patient_id = tf.squeeze(
      features[standard_fields.InputDataFields.patient_id], axis=0)
    exam_id = tf.squeeze(
      features[standard_fields.InputDataFields.examination_name], axis=0)
    image_file = tf.squeeze(
      features[standard_fields.InputDataFields.image_file], axis=0)

    hooks = []
    metric_dict = {}
    lesion_slice_ratio = float(dataset_info[
      standard_fields.PickledDatasetInfo.split_to_num_slices_with_lesion][
        eval_split_name]) / len(dataset_info[
          standard_fields.PickledDatasetInfo.file_names][eval_split_name])
    annotation_mask = tf.cast(tf.squeeze(annotation_mask, axis=-1),
                                tf.float32)
    if pipeline_config.dataset.tfrecords_type == 'input_3d':

      # We are only interested in evaluating the center two slices
      num_slices = scaled_network_output.get_shape().as_list()[0]
      first_slice_index = int(num_slices / 2 - 1)
      scaled_network_output = scaled_network_output[
        first_slice_index:first_slice_index + 2]
      annotation_mask = annotation_mask[
        first_slice_index:first_slice_index + 2]
      image_decoded = image_decoded[
        first_slice_index:first_slice_index + 2]
      slice_ids = slice_ids[first_slice_index: first_slice_index + 2]
      image_file = image_file[first_slice_index: first_slice_index + 2]

    prediction_groundtruth_stack = tf.stack(
      [scaled_network_output, annotation_mask],
      axis=1 if pipeline_config.eval_config.eval_3d_as_2d
      and pipeline_config.dataset.tfrecords_type == 'input_3d' else 0)

    num_froc_thresholds = 200.0
    froc_thresholds = np.linspace(0.0, 1.0, num=num_froc_thresholds,
                                  endpoint=True, dtype=np.float32)

    if pipeline_config.dataset.tfrecords_type == 'input_3d':
      eval_3d_hook = session_hooks.Eval3DHook(
        groundtruth=annotation_mask, prediction=scaled_network_output,
        slice_ids=slice_ids, patient_id=patient_id,
        eval_3d_as_2d=pipeline_config.eval_config.eval_3d_as_2d,
        exam_id=exam_id,
        patient_exam_id_to_num_slices=dataset_info[
          standard_fields.PickledDatasetInfo.patient_exam_id_to_num_slices][
            eval_split_name], calc_froc=calc_froc,
        target_size=(pipeline_config.model.input_image_size_y,
                     pipeline_config.model.input_image_size_x),
        result_folder=result_folder, eval_dir=eval_dir,
        lesion_slice_ratio=lesion_slice_ratio, froc_thresholds=froc_thresholds)

      vis_hook = session_hooks.VisualizationHook(
        result_folder=result_folder,
        visualization_file_names=visualization_file_names,
        file_name=image_file,
        image_decoded=image_decoded,
        annotation_mask=annotation_mask,
        predicted_mask=scaled_network_output, eval_dir=eval_dir,
        is_3d=True)

      hooks.append(eval_3d_hook)
      hooks.append(vis_hook)
    else:
      (metric_dict, statistics_dict, num_lesions, froc_region_cm_values) = (
        metric_utils.get_metrics(
         prediction_groundtruth_stack, parallel_iterations=min(
           pipeline_config.eval_config.batch_size,
           util_ops.get_cpu_count()),
         calc_froc=calc_froc, is_3d=False, thresholds=froc_thresholds))

      vis_hook = session_hooks.VisualizationHook(
        result_folder=result_folder,
        visualization_file_names=visualization_file_names,
        file_name=image_file,
        image_decoded=image_decoded,
        annotation_mask=annotation_mask,
        predicted_mask=scaled_network_output, eval_dir=eval_dir,
        is_3d=False)
      patient_metric_hook = session_hooks.PatientMetricHook(
        statistics_dict=statistics_dict,
        patient_id=patient_id,
        exam_id=exam_id,
        result_folder=result_folder, eval_dir=eval_dir,
        num_lesions=num_lesions,
        froc_region_cm_values=froc_region_cm_values,
        froc_thresholds=froc_thresholds, calc_froc=calc_froc,
        lesion_slice_ratio=lesion_slice_ratio)

      hooks.append(vis_hook)
      hooks.append(patient_metric_hook)

    return tf.estimator.EstimatorSpec(
      mode, loss=total_loss, train_op=train_op,
      evaluation_hooks=hooks,
      eval_metric_ops=metric_dict)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    if (pipeline_config.dataset.tfrecords_type == 'input_3d'):
      assert(False and "Not yet implemented!")
    if pipeline_config.train_config.loss.name == 'sigmoid':
      scaled_network_output = tf.nn.sigmoid(network_output)[:, :, :, 0]
    elif pipeline_config.train_config.loss.name == 'softmax':
      assert(network_output.get_shape().as_list()[-1] == 2)
      scaled_network_output = tf.nn.softmax(network_output)[:, :, :, 1]

    vis_hook = session_hooks.VisualizationHook(
      result_folder=result_folder,
      visualization_file_names=None,
      file_name=features[standard_fields.InputDataFields.image_file],
      image_decoded=features[standard_fields.InputDataFields.image_decoded],
      annotation_decoded=None,
      predicted_mask=scaled_network_output, eval_dir=eval_dir)

    predicted_mask = tf.stack([scaled_network_output * 255,
                               tf.zeros_like(scaled_network_output),
                              tf.zeros_like(scaled_network_output)], axis=3)

    predicted_mask_overlay = tf.clip_by_value(
      features[standard_fields.InputDataFields.image_decoded]
      * 0.5 + predicted_mask, 0, 255)

    return tf.estimator.EstimatorSpec(
      mode, prediction_hooks=[vis_hook], predictions={
        'image_file': features[standard_fields.InputDataFields.image_file],
        'prediction_overlay': predicted_mask_overlay,
        'prediction': predicted_mask})
  else:
    assert(False)


def get_model_fn(pipeline_config, result_folder, dataset_folder, dataset_info,
                 eval_split_name, eval_dir, calc_froc):
  if dataset_info is None:
    visualization_file_names = None
  else:
    file_names = dataset_info[
      standard_fields.PickledDatasetInfo.file_names][eval_split_name]
    np.random.shuffle(file_names)

    patient_ids = dataset_info[
      standard_fields.PickledDatasetInfo.patient_ids][eval_split_name]

    # Select one image per patient
    selected_files = dict()
    for file_name in file_names:
      patient_id = _extract_patient_id(file_name)
      assert(patient_id in patient_ids)
      if patient_id not in selected_files:
        selected_files[patient_id] = file_name

    num_visualizations = pipeline_config.eval_config.num_images_to_visualize
    if num_visualizations is None or num_visualizations == -1:
      num_visualizations = len(selected_files)
    else:
      num_visualizations = min(num_visualizations,
                               len(selected_files))

    visualization_file_names = list(selected_files.values())[
      :num_visualizations]

  model_name = pipeline_config.model.WhichOneof('model_type')
  if model_name == 'unet':
    feature_extractor = unet.UNet(
      weight_decay=pipeline_config.train_config.weight_decay,
      conv_padding=pipeline_config.model.conv_padding,
      filter_sizes=pipeline_config.model.unet.filter_sizes,
      down_activation=pipeline_config.model.unet.down_activation,
      up_activation=pipeline_config.model.unet.up_activation,
      norm_first=pipeline_config.model.norm_first,
      is_3d=pipeline_config.dataset.tfrecords_type == 'input_3d',
      conv_locally_connected=pipeline_config.model.conv_locally_connected)
    return functools.partial(_general_model_fn,
                             pipeline_config=pipeline_config,
                             result_folder=result_folder,
                             dataset_info=dataset_info,
                             feature_extractor=feature_extractor,
                             visualization_file_names=visualization_file_names,
                             eval_dir=eval_dir, calc_froc=calc_froc,
                             as_gan_generator=False,
                             eval_split_name=eval_split_name)
  elif model_name == 'gan':
    generator_name = pipeline_config.model.gan.WhichOneof('generator')
    discriminator_name = pipeline_config.model.gan.WhichOneof('discriminator')

    if generator_name == 'unetp2p':
      generator = unet_pix2pix.UNetP2P(
        conv_bn_first=pipeline_config.model.conv_bn_first)
    else:
      assert(False)

    if discriminator_name == 'gan_discriminator':
      discriminator = gan_discriminator.GANDiscriminator(
        conv_bn_first=pipeline_config.model.conv_bn_first)
    else:
      assert(False)

    generator_model_fn = functools.partial(
      _general_model_fn,
      pipeline_config=pipeline_config,
      result_folder=result_folder,
      dataset_info=dataset_info,
      feature_extractor=generator,
      visualization_file_names=visualization_file_names,
      eval_dir=eval_dir, calc_froc=calc_froc,
      as_gan_generator=True, eval_split_name=eval_split_name)
    discriminator_model_fn = functools.partial(
      _gan_discriminator_model_fn, model=discriminator,
      use_batch_norm=pipeline_config.model.use_batch_norm,
      bn_momentum=pipeline_config.model.batch_norm_momentum,
      bn_epsilon=pipeline_config.model.batch_norm_epsilon)

    return generator_model_fn, discriminator_model_fn
  else:
    assert(False)
