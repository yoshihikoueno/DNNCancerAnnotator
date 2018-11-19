import logging
import os
import sys
import pdb
from datetime import datetime
from shutil import copy

import tensorflow as tf

from utils import standard_fields
from utils import util_ops
from utils import setup_utils
from builders import estimator_builder

tf.logging.set_verbosity(tf.logging.INFO)

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

FLAGS = flags.FLAGS


def main(_):
  if FLAGS.pdb:
    debugger = pdb.Pdb(stdout=sys.__stdout__)
    debugger.set_trace()

  if not os.path.isdir(FLAGS.result_dir):
    raise ValueError("Result directory does not exist!")

  if FLAGS.warm_start:
    pipeline_config_file = FLAGS.result_dir, 'pipeline_config'
  else:
    pipeline_config_file = FLAGS.pipeline_config_file

  pipeline_config = setup_utils.load_config(pipeline_config_file)

  if FLAGS.warm_start:
    existing_tfrecords_directory = FLAGS.result_dir
  else:
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
    existing_tfrecords_directory = None

  util_ops.init_logger(FLAGS.result_dir, FLAGS.warm_start)

  logging.info("Command line arguments: {}".format(sys.argv))

  input_fn = setup_utils.get_input_fn(
    pipeline_config=pipeline_config, directory=FLAGS.result_dir,
    existing_tfrecords_directory=existing_tfrecords_directory,
    split_name=standard_fields.SplitNames.train)

  num_gpu = pipeline_config.train_config.num_gpu
  real_gpu_nb = len(util_ops.get_devices())
  if num_gpu and num_gpu > real_gpu_nb:
    raise ValueError("Too many GPUs specified!")
  else:
    num_gpu = real_gpu_nb

  distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpu)

  estimator = estimator_builder.build_estimator(
    pipeline_config=pipeline_config, result_dir=FLAGS.result_dir,
    warm_start_path=existing_tfrecords_directory,
    train_distribution=distribution,
    eval_distribution=None, warm_start_ckpt_name=None)

  estimator.train(input_fn=input_fn,
                  max_steps=pipeline_config.train_config.num_steps)


if __name__ == '__main__':
  tf.app.run()
