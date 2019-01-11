import logging
import os
import sys
import pdb
from datetime import datetime
from shutil import copy

import numpy as np
import tensorflow as tf

from utils import standard_fields
from utils import util_ops
from utils import setup_utils
from builders import estimator_builder

flags = tf.app.flags
flags.DEFINE_string('pipeline_config_file', '',
                    'Path to the pipeline config file. If resume is true,'
                    ' will take pipeline config from result_dir instead.')
flags.DEFINE_string('result_dir', '', 'Path to the folder, in which to'
                                      'create the result folder.')
flags.DEFINE_string('prefix', '', 'An optional prefix for the result'
                                  'folder name.')
flags.DEFINE_bool('warm_start', False, 'Whether to resume training from a '
                                   'previous checkpoint. If true, there'
                                   'will be no new result folder created')
flags.DEFINE_bool('pdb', False, 'Whether to use pdb debugging functionality.')
flags.DEFINE_integer('num_gpu', -1, 'Number of GPUs to use. '
                     '-1 to use all available')
flags.DEFINE_integer('visible_device_index', -1, 'Index of the visible device')
flags.DEFINE_integer('num_train_steps', -1, 'Number of max training steps')
flags.DEFINE_integer('num_eval_steps', -1, 'Number of max eval steps')

FLAGS = flags.FLAGS


def main(_):
  assert(FLAGS.num_train_steps > 0)
  if FLAGS.pdb:
    debugger = pdb.Pdb(stdout=sys.__stdout__)
    debugger.set_trace()

  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  if FLAGS.num_gpu == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
  else:
    if FLAGS.visible_device_index != -1:
      assert(FLAGS.num_gpu == 1)
      os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(
        FLAGS.visible_device_index)

  if not os.path.isdir(FLAGS.result_dir):
    raise ValueError("Result directory does not exist!")

  if FLAGS.warm_start:
    pipeline_config_file = FLAGS.result_dir, 'pipeline_config'
  else:
    pipeline_config_file = FLAGS.pipeline_config_file

  pipeline_config = setup_utils.load_config(pipeline_config_file)

  np.random.seed(pipeline_config.seed)
  tf.set_random_seed(pipeline_config.seed)

  if not FLAGS.warm_start:
    if FLAGS.prefix:
      FLAGS.prefix += '_'
    # Create the new result folder.
    result_folder_name = FLAGS.prefix + 'train_{}_{}'.format(
      os.path.basename(pipeline_config.dataset.dataset_path),
      datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    FLAGS.result_dir = os.path.join(FLAGS.result_dir, result_folder_name)
    os.mkdir(FLAGS.result_dir)
    copy(pipeline_config_file, os.path.join(FLAGS.result_dir,
                                            'pipeline.config'))

  util_ops.init_logger(FLAGS.result_dir, FLAGS.warm_start)

  logging.info("Command line arguments: {}".format(sys.argv))

  num_gpu = FLAGS.num_gpu
  if num_gpu <= -1:
    num_gpu = None
  real_gpu_nb = len(util_ops.get_devices())
  if num_gpu is not None:
    if num_gpu > real_gpu_nb:
      raise ValueError("Too many GPUs specified!")
  else:
    num_gpu = real_gpu_nb

  if num_gpu > 1:
    train_distribution = tf.contrib.distribute.MirroredStrategy(
      num_gpus=num_gpu)
  else:
    train_distribution = None

  # For now we cannot have mirrored evaluation distribution
  eval_distribution = None

  train_input_fn, dataset_info = setup_utils.get_input_fn(
    pipeline_config=pipeline_config, directory=FLAGS.result_dir,
    existing_tfrecords=FLAGS.warm_start,
    split_name=standard_fields.SplitNames.train,
    is_training=True)

  estimator = estimator_builder.build_estimator(
    pipeline_config=pipeline_config, result_dir=FLAGS.result_dir,
    dataset_info=dataset_info,
    eval_split_name=standard_fields.SplitNames.val,
    warm_start_path=FLAGS.result_dir if FLAGS.warm_start else None,
    train_distribution=train_distribution,
    eval_distribution=eval_distribution, num_gpu=num_gpu,
    warm_start_ckpt_name=None)

  if pipeline_config.train_config.early_stopping:
    early_stop_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
      estimator=estimator, metric_name='loss',
      max_steps_without_decrease=pipeline_config.train_config.
      early_stopping_max_steps_without_decrease,
      min_steps=pipeline_config.train_config.early_stopping_min_steps,
      run_every_secs=pipeline_config.train_config.
      early_stopping_run_every_secs)
  else:
    early_stop_hook = None

  train_spec = tf.estimator.TrainSpec(
    input_fn=train_input_fn, max_steps=FLAGS.num_train_steps,
    hooks=early_stop_hook if early_stop_hook is None else [early_stop_hook])

  eval_input_fn, _ = setup_utils.get_input_fn(
    pipeline_config=pipeline_config, directory=FLAGS.result_dir,
    existing_tfrecords=True, split_name=standard_fields.SplitNames.val,
    is_training=False)

  eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input_fn,
    steps=None if FLAGS.num_eval_steps <= 0 else FLAGS.num_eval_steps,
    start_delay_secs=0, throttle_secs=0, name='val_eval')

  tf.estimator.train_and_evaluate(estimator=estimator,
                                  train_spec=train_spec, eval_spec=eval_spec)

  # Evaluate train set
  train_eval_input_fn, _ = setup_utils.get_input_fn(
    pipeline_config=pipeline_config, directory=FLAGS.result_dir,
    existing_tfrecords=True,
    split_name=standard_fields.SplitNames.train,
    is_training=False)

  tf.estimator.evaluate(input_fn=train_eval_input_fn, steps=None,
                        name='train_eval')


if __name__ == '__main__':
  tf.app.run()
