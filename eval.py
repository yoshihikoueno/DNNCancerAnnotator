import pdb
import sys
import os
import logging
import time

import numpy as np
import tensorflow as tf

from utils import setup_utils
from utils import util_ops
from utils import standard_fields
from builders import estimator_builder

flags = tf.app.flags
flags.DEFINE_string('checkpoint_dir', '', 'The checkpoint directory to load '
                                          'the network from. If '
                                          'checkpoint_name is not '
                                          'defined it will use the '
                                          'latest checkpoint file in this '
                                          'directory')
flags.DEFINE_string('checkpoint_name', '', 'Optional name of a specific '
                                        'checkpoint')
flags.DEFINE_bool('all_checkpoints', False,
                  'Optional flag to evaluate all existing checkpoints in '
                  'checkpoint_dir from the beginning on.')
flags.DEFINE_bool('continuous', False, 'Whether to keep the process alive and '
                  'query for new checkpoints at a certain interval')
flags.DEFINE_string('split_name', '', 'train, val, or test')
flags.DEFINE_bool('pdb', False, 'Whether to use pdb debugging functionality.')
flags.DEFINE_bool('use_gpu', False, 'Whether to use GPU')
flags.DEFINE_integer('num_steps', 0,
                     'For debugging purposes, a possible limit to '
                     'the number of steps. 0 means no limit')
flags.DEFINE_integer('visible_device_index', -1, 'Index of the visible device')

FLAGS = flags.FLAGS


def _eval_checkpoint(checkpoint_path, estimator, input_fn, num_steps,
                     split_name):
  logging.info('Evaluating {}'.format(checkpoint_path))
  return estimator.evaluate(input_fn=input_fn, checkpoint_path=checkpoint_path,
                            steps=num_steps, name=split_name)


def main(_):
  assert(not (FLAGS.continuous and FLAGS.checkpoint_name))
  assert(not (FLAGS.checkpoint_name != '' and FLAGS.all_checkpoints))
  assert(FLAGS.split_name in standard_fields.SplitNames.available_names)
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
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

  if not os.path.isdir(FLAGS.checkpoint_dir):
    raise ValueError("Checkpoint directory does not exist!")

  pipeline_config_file = os.path.join(FLAGS.checkpoint_dir,
                                      'pipeline.config')
  pipeline_config = setup_utils.load_config(pipeline_config_file)

  num_parallel_iterations = util_ops.get_cpu_count()

  np.random.seed(pipeline_config.seed)
  tf.set_random_seed(pipeline_config.seed)

  result_folder = os.path.join(FLAGS.checkpoint_dir,
                               'eval_{}'.format(FLAGS.split_name))
  if os.path.exists(result_folder):
    raise ValueError("Evaluation already took place!")

  os.mkdir(result_folder)

  # Init Logger
  util_ops.init_logger(result_folder)
  logging.info("Command line arguments: {}".format(sys.argv))

  if num_gpu > 1:
    device_strings = ['/device:GPU:{}'.format(index)
                      for index in range(num_gpu)]
    distribution = tf.distribute.MirroredStrategy(
      devices=device_strings)
  else:
    distribution = None

  input_fn, dataset_info = setup_utils.get_input_fn(
    pipeline_config=pipeline_config, directory=FLAGS.checkpoint_dir,
    existing_tfrecords=True,
    split_name=FLAGS.split_name, is_training=False,
    num_parallel_iterations=num_parallel_iterations)

  estimator = estimator_builder.build_estimator(
    pipeline_config=pipeline_config, checkpoint_folder=FLAGS.checkpoint_dir,
    dataset_info=dataset_info,
    dataset_folder=pipeline_config.dataset.dataset_path,
    eval_split_name=FLAGS.split_name, train_distribution=None,
    eval_distribution=distribution,
    eval_dir=result_folder,
    calc_froc=not FLAGS.all_checkpoints and not FLAGS.continuous)

  if FLAGS.num_steps > 0:
    num_steps = FLAGS.num_steps
  else:
    num_steps = None

  if FLAGS.checkpoint_name:
    _eval_checkpoint(
      os.path.join(FLAGS.checkpoint_dir,
                   FLAGS.checkpoint_name), estimator, input_fn, num_steps,
      split_name=FLAGS.split_name)
  elif FLAGS.all_checkpoints:
    latest_checkpoint = ''
    evaluated_checkpoints = []
    all_checkpoints = []
    # We need a loop, as new checkpoints might be generated while we are still
    # evaluating
    while True:
      all_checkpoints = tf.train.get_checkpoint_state(
        FLAGS.checkpoint_dir).all_model_checkpoint_paths
      new_checkpoint_evaluated = False
      for checkpoint in all_checkpoints:
        if checkpoint in evaluated_checkpoints:
          continue
        else:
          new_checkpoint_evaluated = True
        _eval_checkpoint(checkpoint, estimator, input_fn, num_steps,
                         split_name=FLAGS.split_name)
        evaluated_checkpoints.append(checkpoint)
        latest_checkpoint = checkpoint
      if not new_checkpoint_evaluated:
        if FLAGS.continuous:
          logging.info('No new checkpoint. Checking back in {} seconds.'
                       .format(pipeline_config.eval_config.eval_interval_secs))
          time.sleep(pipeline_config.eval_config.eval_interval_secs)
        else:
          break

    # Reevaluate last checkpoint to calculate froc
    estimator = estimator_builder.build_estimator(
      pipeline_config=pipeline_config, checkpoint_folder=FLAGS.checkpoint_dir,
      dataset_info=dataset_info,
      dataset_folder=pipeline_config.dataset.dataset_path,
      eval_split_name=FLAGS.split_name, train_distribution=None,
      eval_distribution=distribution,
      eval_dir=result_folder,
      calc_froc=True)

    ckpt = all_checkpoints[-1]

    _eval_checkpoint(ckpt, estimator, input_fn, num_steps,
                     split_name=FLAGS.split_name)
  elif FLAGS.continuous:
    latest_checkpoint = ''
    while True:
      next_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

      if not next_checkpoint or next_checkpoint == latest_checkpoint:
        logging.info('No new checkpoint. Checking back in {} seconds.'.format(
          pipeline_config.eval_config.eval_interval_secs))
        time.sleep(pipeline_config.eval_config.eval_interval_secs)
        continue

      _eval_checkpoint(next_checkpoint, estimator, input_fn, num_steps,
                       split_name=FLAGS.split_name)

      latest_checkpoint = next_checkpoint
  else:
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    _eval_checkpoint(latest_checkpoint, estimator, input_fn, num_steps,
                     split_name=FLAGS.split_name)


if __name__ == '__main__':
  tf.app.run()
