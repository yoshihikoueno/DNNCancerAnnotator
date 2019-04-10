import numpy as np
import tensorflow as tf

from builders import model_builder


def build_estimator(pipeline_config, result_dir, dataset_folder, dataset_info,
                    eval_split_name, train_distribution, eval_distribution,
                    num_gpu, eval_dir):
  np.random.seed(pipeline_config.seed)
  tf.set_random_seed(pipeline_config.seed)

  model_fn = model_builder.get_model_fn(pipeline_config=pipeline_config,
                                        result_folder=result_dir,
                                        dataset_folder=dataset_folder,
                                        dataset_info=dataset_info,
                                        eval_split_name=eval_split_name,
                                        num_gpu=num_gpu, eval_dir=eval_dir)

  # keep_checkpoint_max is set to this number because when setting it as no
  # limit i.e. 0 or None, the checkpoint file will not contain all checkpoint
  # names.
  run_config = tf.estimator.RunConfig(
    model_dir=result_dir, tf_random_seed=pipeline_config.seed,
    save_summary_steps=pipeline_config.train_config.save_summary_steps,
    save_checkpoints_secs=pipeline_config.train_config.save_checkpoints_secs,
    session_config=tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False),
    keep_checkpoint_max=9999999, log_step_count_steps=10,
    train_distribute=train_distribution, eval_distribute=eval_distribution)

  estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

  return estimator
