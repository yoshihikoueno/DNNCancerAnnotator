import pdb
import sys
import os
import logging
import time
from datetime import datetime
from shutil import copy

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
flags.DEFINE_string('result_dir', '', 'Optional directory to write the '
                                      'results to.')
flags.DEFINE_bool('use_gpu', False, 'Whether to use GPU or CPU')
flags.DEFINE_bool('repeated', False, 'Whether to evaluate successive '
                                     'checkpoints')
flags.DEFINE_string('split_name', '', 'train, val, or test')
flags.DEFINE_bool('pdb', False, 'Whether to use pdb debugging functionality.')

FLAGS = flags.FLAGS


def main(_):
  assert(not (FLAGS.repeated and FLAGS.checkpoint_name))
  if not FLAGS.use_gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

  assert(FLAGS.split_name in standard_fields.SplitNames.available_names)
  if FLAGS.pdb:
    debugger = pdb.Pdb(stdout=sys.__stdout__)
    debugger.set_trace()

  if not os.path.isdir(FLAGS.checkpoint_dir):
    raise ValueError("Checkpoint directory does not exist!")

  pipeline_config_file = os.path.join(FLAGS.checkpoint_dir,
                                      'pipeline.config')
  pipeline_config = setup_utils.load_config(pipeline_config_file)

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

  num_gpu = pipeline_config.train_config.num_gpu
  real_gpu_nb = len(util_ops.get_devices())
  if num_gpu and num_gpu > real_gpu_nb:
    raise ValueError("Too many GPUs specified!")
  else:
    num_gpu = real_gpu_nb

  input_fn, dataset_info = setup_utils.get_input_fn(
    pipeline_config=pipeline_config, directory=FLAGS.checkpoint_dir,
    existing_tfrecords=True,
    split_name=FLAGS.split_name, num_gpu=num_gpu, is_training=False)

  estimator = estimator_builder.build_estimator(
    pipeline_config=pipeline_config, result_dir=result_folder,
    dataset_info=dataset_info,
    dataset_split_name=FLAGS.split_name,
    warm_start_path=FLAGS.checkpoint_dir, train_distribution=None,
    eval_distribution=None, warm_start_ckpt_name=FLAGS.checkpoint_name)

  if FLAGS.checkpoint_name:
    latest_checkpoint = os.path.join(FLAGS.checkpoint_dir,
                                     FLAGS.checkpoint_name)

    estimator.eval(input_fn=input_fn, checkpoint_path=latest_checkpoint)

  elif FLAGS.repeated:
    last_checkpoint = ''
    while True:
      latest_checkpoint = os.path.join(
        FLAGS.checkpoint_dir, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
      if last_checkpoint == latest_checkpoint:
        logging.info('No new checkpoint. Checking back in {} seconds.'.format(
          pipeline_config.eval_config.eval_interval_secs))
        time.sleep(pipeline_config.eval_config.eval_interval_secs)
        continue

      estimator.eval(input_fn=input_fn, checkpoint_path=latest_checkpoint)

      last_checkpoint = latest_checkpoint
  else:
    latest_checkpoint = os.path.join(
      FLAGS.checkpoint_dir, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

    estimator.evaluate(input_fn=input_fn, checkpoint_path=latest_checkpoint)


if __name__ == '__main__':
  tf.app.run()
