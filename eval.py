import pdb
import sys
import os
import logging
import time
from datetime import datetime
from shutil import copy

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
flags.DEFINE_string('result_dir', '', 'Optional directory to write the '
                                      'results to.')
flags.DEFINE_bool('repeated', False, 'Whether to evaluate successive '
                                     'checkpoints')
flags.DEFINE_string('split_name', '', 'train, val, or test')
flags.DEFINE_bool('pdb', False, 'Whether to use pdb debugging functionality.')
flags.DEFINE_integer('num_steps', 0,
                     'For debugging purposes, a possible limit to '
                     'the number of steps. 0 means no limit')

FLAGS = flags.FLAGS


def main(_):
  assert(not (FLAGS.repeated and FLAGS.checkpoint_name))
  assert(not (FLAGS.checkpoint_name != '' and FLAGS.all_checkpoints))

  assert(FLAGS.split_name in standard_fields.SplitNames.available_names)
  if FLAGS.pdb:
    debugger = pdb.Pdb(stdout=sys.__stdout__)
    debugger.set_trace()

  if not os.path.isdir(FLAGS.checkpoint_dir):
    raise ValueError("Checkpoint directory does not exist!")

  pipeline_config_file = os.path.join(FLAGS.checkpoint_dir,
                                      'pipeline.config')
  pipeline_config = setup_utils.load_config(pipeline_config_file)

  np.random.seed(pipeline_config.seed)
  tf.set_random_seed(pipeline_config.seed)

  if pipeline_config.eval_config.num_gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

  result_folder_name = 'eval_{}_{}_{}'.format(
    FLAGS.split_name + '-split',
    os.path.basename(pipeline_config.dataset.dataset_path),
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

  if FLAGS.result_dir:
    result_folder = os.path.join(FLAGS.result_dir, result_folder_name)
  else:
    result_folder = os.path.join(FLAGS.checkpoint_dir, result_folder_name)
  os.mkdir(result_folder)
  copy(pipeline_config_file, os.path.join(result_folder,
                                                'pipeline.config'))

  # Init Logger
  util_ops.init_logger(result_folder)
  logging.info("Command line arguments: {}".format(sys.argv))

  num_gpu = pipeline_config.eval_config.num_gpu
  if num_gpu <= -1:
    num_gpu = None
  real_gpu_nb = len(util_ops.get_devices())
  if num_gpu is not None:
    if num_gpu > real_gpu_nb:
      raise ValueError("Too many GPUs specified!")
  else:
    num_gpu = real_gpu_nb

  input_fn, dataset_info = setup_utils.get_input_fn(
    pipeline_config=pipeline_config, directory=FLAGS.checkpoint_dir,
    existing_tfrecords=True,
    split_name=FLAGS.split_name, is_training=False)

  estimator = estimator_builder.build_estimator(
    pipeline_config=pipeline_config, result_dir=result_folder,
    dataset_info=dataset_info,
    dataset_split_name=FLAGS.split_name,
    warm_start_path=FLAGS.checkpoint_dir, train_distribution=None,
    eval_distribution=None, num_gpu=num_gpu,
    warm_start_ckpt_name=FLAGS.checkpoint_name)

  if FLAGS.num_steps > 0:
    num_steps = FLAGS.num_steps
  else:
    num_steps = None

  if FLAGS.all_checkpoints:
    all_checkpoints = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

  if FLAGS.checkpoint_name:
    latest_checkpoint = os.path.join(FLAGS.checkpoint_dir,
                                     FLAGS.checkpoint_name)

    estimator.evaluate(input_fn=input_fn, checkpoint_path=latest_checkpoint,
                       steps=num_steps)

  elif FLAGS.repeated:
    last_checkpoint = ''
    while True:
      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

      if not latest_checkpoint or last_checkpoint == latest_checkpoint:
        logging.info('No new checkpoint. Checking back in {} seconds.'.format(
          pipeline_config.eval_config.eval_interval_secs))
        time.sleep(pipeline_config.eval_config.eval_interval_secs)
        continue

      logging.info('Evaluating {}'.format(latest_checkpoint))
      estimator.evaluate(input_fn=input_fn, checkpoint_path=latest_checkpoint,
                         steps=num_steps)

      last_checkpoint = latest_checkpoint
  else:
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    estimator.evaluate(input_fn=input_fn, checkpoint_path=latest_checkpoint,
                       steps=num_steps)


if __name__ == '__main__':
  tf.app.run()
