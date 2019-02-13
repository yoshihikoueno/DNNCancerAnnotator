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
flags.DEFINE_bool('resume', False, 'Whether to resume training from a '
                                   'previous checkpoint. If true, there'
                                   'will be no new result folder created')
flags.DEFINE_bool('pdb', False, 'Whether to use pdb debugging functionality.')
flags.DEFINE_bool('use_gpu', True, 'Whether to use GPU')
flags.DEFINE_integer('num_train_steps', -1, 'Number of max training steps')
flags.DEFINE_integer('num_eval_steps', -1, 'Number of max eval steps')

FLAGS = flags.FLAGS


def main(_):
  assert(FLAGS.num_train_steps > 0)
  if FLAGS.pdb:
    debugger = pdb.Pdb(stdout=sys.__stdout__)
    debugger.set_trace()

  # Get the number of GPU from env
  num_gpu = 0
  if FLAGS.use_gpu:
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
      index_list = os.environ['CUDA_VISIBLE_DEVICES']
      index_list = index_list.split(',')
      if len(index_list) == 1 and index_list[0] == '':
        # No indices set, use all GPUs available
        num_gpu = len(util_ops.get_devices())
      else:
        num_gpu = len(index_list)
    else:
      num_gpu = len(util_ops.get_devices())

  if not os.path.isdir(FLAGS.result_dir):
    raise ValueError("Result directory does not exist!")

  if FLAGS.resume:
    pipeline_config_file = os.path.join(FLAGS.result_dir, 'pipeline.config')
  else:
    pipeline_config_file = FLAGS.pipeline_config_file

  pipeline_config = setup_utils.load_config(pipeline_config_file)

  np.random.seed(pipeline_config.seed)
  tf.set_random_seed(pipeline_config.seed)

  if not FLAGS.resume:
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

  util_ops.init_logger(FLAGS.result_dir, FLAGS.resume)

  logging.info("Command line arguments: {}".format(sys.argv))

  if num_gpu > 1:
    train_distribution = tf.contrib.distribute.MirroredStrategy(
      num_gpus=num_gpu)
  else:
    train_distribution = None

  # For now we cannot have mirrored evaluation distribution
  eval_distribution = None

  train_input_fn, dataset_info = setup_utils.get_input_fn(
    pipeline_config=pipeline_config, directory=FLAGS.result_dir,
    existing_tfrecords=FLAGS.resume,
    split_name=standard_fields.SplitNames.train,
    is_training=True)

  estimator = estimator_builder.build_estimator(
    pipeline_config=pipeline_config, result_dir=FLAGS.result_dir,
    dataset_info=dataset_info,
    eval_split_name=standard_fields.SplitNames.val,
    train_distribution=train_distribution,
    eval_distribution=eval_distribution, num_gpu=num_gpu)

  if pipeline_config.train_config.early_stopping:
    early_stop_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
      estimator=estimator, metric_name='loss',
      max_steps_without_decrease=pipeline_config.
      train_config.early_stopping_max_steps_without_decrease,
      min_steps=pipeline_config.train_config.early_stopping_min_steps,
      run_every_secs=pipeline_config.train_config.
      early_stopping_run_every_secs,
      eval_dir=os.path.join(FLAGS.result_dir,
                            'eval_' + standard_fields.SplitNames.val))
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
    start_delay_secs=0, throttle_secs=0, name=standard_fields.SplitNames.val)

  tf.estimator.train_and_evaluate(estimator=estimator,
                                  train_spec=train_spec, eval_spec=eval_spec)

  # Evaluate train set
  train_eval_input_fn, _ = setup_utils.get_input_fn(
    pipeline_config=pipeline_config, directory=FLAGS.result_dir,
    existing_tfrecords=True,
    split_name=standard_fields.SplitNames.train,
    is_training=False)

  estimator = estimator_builder.build_estimator(
    pipeline_config=pipeline_config, result_dir=FLAGS.result_dir,
    dataset_info=dataset_info,
    eval_split_name=standard_fields.SplitNames.train,
    train_distribution=train_distribution,
    eval_distribution=eval_distribution, num_gpu=num_gpu)

  estimator.evaluate(input_fn=train_eval_input_fn, steps=None,
                     name=standard_fields.SplitNames.train)


if __name__ == '__main__':
  tf.app.run()
