import pdb
import sys
import os
import logging
import time

from PIL import Image
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
flags.DEFINE_string('input_dir', '',
                    'Directory containing examples to predict')
flags.DEFINE_bool('pdb', False, 'Whether to use pdb debugging functionality.')
flags.DEFINE_bool('use_gpu', False, 'Whether to use GPU')
flags.DEFINE_integer('visible_device_index', -1, 'Index of the visible device')

FLAGS = flags.FLAGS


def main(_):
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
  if not os.path.isdir(FLAGS.input_dir):
    raise ValueError("Input directory does not exist!")

  pipeline_config_file = os.path.join(FLAGS.checkpoint_dir,
                                      'pipeline.config')
  pipeline_config = setup_utils.load_config(pipeline_config_file)

  np.random.seed(pipeline_config.seed)
  tf.set_random_seed(pipeline_config.seed)

  # Init Logger
  result_folder = os.path.join(FLAGS.input_dir, 'predict_result')
  if os.path.exists(result_folder):
    raise ValueError("Prediction already occured!")

  os.mkdir(result_folder)
  util_ops.init_logger(result_folder)
  logging.info("Command line arguments: {}".format(sys.argv))

  if num_gpu > 1:
    device_strings = ['/device:GPU:{}'.format(index)
                      for index in range(num_gpu)]
    distribution = tf.distribute.MirroredStrategy(
      devices=device_strings)
  else:
    distribution = None

  input_fn = setup_utils.get_predict_input_fn(pipeline_config, FLAGS.input_dir)

  estimator = estimator_builder.build_estimator(
    pipeline_config=pipeline_config, checkpoint_folder=FLAGS.checkpoint_dir,
    dataset_info=None, eval_split_name=None, train_distribution=None,
    dataset_folder=None,
    eval_distribution=distribution, num_gpu=num_gpu,
    eval_dir=result_folder, calc_froc=False)

  predictions = estimator.predict(
    input_fn, yield_single_examples=False)
  overlay_folder = os.path.join(result_folder, 'prediction_overlay')
  prediction_folder = os.path.join(result_folder, 'prediction')

  os.mkdir(overlay_folder)
  os.mkdir(prediction_folder)

  for prediction in predictions:
    file_name = os.path.basename(prediction['image_file'][0].decode('UTF-8'))
    prediction_overlay = prediction['prediction_overlay'][0]
    prediction = prediction['prediction'][0]

    prediction_overlay = prediction_overlay.astype(np.uint8)
    prediction = prediction.astype(np.uint8)

    prediction_overlay_img = Image.fromarray(prediction_overlay)
    prediction_overlay_img.save(os.path.join(overlay_folder, file_name))

    prediction_img = Image.fromarray(prediction)
    prediction_img.save(os.path.join(prediction_folder, file_name))


if __name__ == '__main__':
  tf.app.run()
