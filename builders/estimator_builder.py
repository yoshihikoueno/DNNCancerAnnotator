import functools

import numpy as np
import tensorflow as tf
tfgan = tf.contrib.gan

from builders import model_builder
from builders import optimizer_builder


def build_estimator(pipeline_config, checkpoint_folder,
                    dataset_folder, dataset_info,
                    eval_split_name, train_distribution, eval_distribution,
                    eval_dir, calc_froc):
  np.random.seed(pipeline_config.seed)
  tf.set_random_seed(pipeline_config.seed)

  # keep_checkpoint_max is set to this number because when setting it as no
  # limit i.e. 0 or None, the checkpoint file will not contain all checkpoint
  # names.
  run_config = tf.estimator.RunConfig(
    model_dir=checkpoint_folder, tf_random_seed=pipeline_config.seed,
    save_summary_steps=pipeline_config.train_config.save_summary_steps,
    save_checkpoints_secs=pipeline_config.train_config.save_checkpoints_secs,
    session_config=tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False),
    keep_checkpoint_max=9999999, log_step_count_steps=10,
    train_distribute=train_distribution, eval_distribute=eval_distribution)

  is_gan_model = pipeline_config.model.WhichOneof('model_type') == 'gan'
  if is_gan_model:
    generator_fn, discriminator_fn = model_builder.get_model_fn(
      pipeline_config=pipeline_config, result_folder=checkpoint_folder,
      dataset_folder=dataset_folder, dataset_info=dataset_info,
      eval_split_name=eval_split_name, eval_dir=eval_dir, calc_froc=calc_froc)

    generator_loss = functools.partial(tfgan.losses.modified_generator_loss,
                                       add_summaries=True)
    discriminator_loss = functools.partial(
      tfgan.losses.modified_discriminator_loss, add_summaries=True)

    optimizer = optimizer_builder.build(
      pipeline_config.train_config.optimizer)[0]

    gan_estimator = tfgan.estimator.GANEstimator(
      model_dir=checkpoint_folder, generator_fn=generator_fn,
      discriminator_fn=discriminator_fn, generator_loss_fn=generator_loss,
      discriminator_loss_fn=discriminator_loss, generator_optimizer=optimizer,
      discriminator_optimizer=optimizer, config=run_config,
      add_summaries=tfgan.estimator.SummaryType.IMAGE_COMPARISON,
      use_loss_summaries=True)

    return gan_estimator
  else:
    model_fn = model_builder.get_model_fn(pipeline_config=pipeline_config,
                                          result_folder=checkpoint_folder,
                                          dataset_folder=dataset_folder,
                                          dataset_info=dataset_info,
                                          eval_split_name=eval_split_name,
                                          eval_dir=eval_dir,
                                          calc_froc=calc_froc)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

    return estimator
