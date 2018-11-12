import logging
import os
import sys
import pickle
import pdb
from datetime import datetime
from shutil import copy

import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from utils import standard_fields
from utils import util_ops
from dataset_helpers import prostate_cancer_utils as pc
import trainer
from protos import pipeline_pb2

tf.logging.set_verbosity(tf.logging.INFO)

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
flags.DEFINE_bool('use_tfrecords', True,
                  'Whether to create a tfrecords file from the dataset for'
                  'efficient streaming or just raw access to the dataset files.')
flags.DEFINE_bool('use_existing_tfrecords', False,
                  'Whether to use an existing tfrecords file in the dataset folder,'
                  'or to create one according to the pipeline config')
flags.DEFINE_bool('pdb', False, 'Whether to use pdb debugging functionality.')

FLAGS = flags.FLAGS


def _get_configs_from_pipeline_file(pipeline_config_path):
  """Reads configuration from a pipeline_pb2.PipelineConfig.
  Args:
    pipeline_config_path: Path to pipeline_pb2.PipelineConfig text
      proto.
  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`. Value are the
      corresponding config objects.
  """
  pipeline_config = pipeline_pb2.PipelineConfig()
  with open(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  return pipeline_config


def _load_config(pipeline_config_file, result_dir, resume):
  if resume:
    # Take the pipeline config file from within result_dir
    pipeline_config_file = os.path.join(result_dir, 'pipeline.config')
    pipeline_config = _get_configs_from_pipeline_file(
      pipeline_config_file)
  else:
    # Take the pipeline config file from pipeline_config_file
    if not os.path.isfile(pipeline_config_file):
      raise ValueError("Invalid pipeline config file specified!")
    pipeline_config = _get_configs_from_pipeline_file(
      pipeline_config_file)

  return pipeline_config, pipeline_config_file


def _load_existing_tfrecords(directory, split_name, target_dims):
  tfrecords_file = os.path.join(directory, split_name + '.tfrecords')
  meta_file = os.path.join(
    directory, standard_fields.PickledDatasetInfo.pickled_file_name)

  f = open(meta_file, 'rb')
  meta_data = pickle.load(f)
  f.close()

  if split_name == standard_fields.SplitNames.train:
    ids = meta_data[standard_fields.PickledDatasetInfo.train_patient_ids]
  elif split_name == standard_fields.SplitNames.val:
    ids = meta_data[standard_fields.PickledDatasetInfo.val_patient_ids]
  elif split_name == standard_fields.SplitNames.test:
    ids = meta_data[standard_fields.PickledDatasetInfo.test_patient_ids]
  else:
    assert(False)

  return pc.build_tf_dataset_from_tfrecords(
    tfrecords_file, target_dims, ids,
    split_name == standard_fields.SplitNames.train), meta_data


def _prepare_dataset(result_dir, config_file, pipeline_config,
                     prefix, use_existing_tfrecords,
                     use_tfrecords):
  target_dims = [pipeline_config.model.input_image_size_x,
                 pipeline_config.model.input_image_size_y,
                 pipeline_config.model.input_image_channels]

  copy(config_file, os.path.join(result_dir, 'pipeline.config'))
  if use_existing_tfrecords:
      for split in standard_fields.SplitNames.available_names:
        copy(os.path.join(pipeline_config.dataset.dataset_path,
                          split + '.tfrecords'),
             result_dir)
      copy(os.path.join(pipeline_config.dataset.dataset_path,
                        standard_fields.PickledDatasetInfo.pickled_file_name),
           result_dir)

      dataset, meta_info = _load_existing_tfrecords(
        result_dir, standard_fields.SplitNames.train, target_dims)
  else:
    if use_tfrecords:
      input_dims = [pipeline_config.model.input_image_size_x,
                    pipeline_config.model.input_image_size_y,
                    pipeline_config.model.input_image_channels]
      # We need to create a tfrecords, and use it as dataset

      # Tfrecords files and meta file are created in result dir
      pc.build_tfrecords_from_files(pipeline_config.dataset, input_dims,
                                    pipeline_config.seed, result_dir)

      # Load tfrecords into dataset
      dataset, meta_info = _load_existing_tfrecords(
        result_dir, standard_fields.SplitNames.train, target_dims)
    else:
      # We just need to create a tf dataset
      dataset, _, _, meta_info = \
        pc.build_tf_dataset_from_files(pipeline_config.dataset, input_dims,
                                       pipeline_config.seed)

  dataset = dataset.batch(pipeline_config.train_config.batch_size)

  dataset = dataset.prefetch(pipeline_config.train_config.batch_size)

  return dataset


def main(_):
  if FLAGS.pdb:
    debugger = pdb.Pdb(stdout=sys.__stdout__)
    debugger.set_trace()
  if not os.path.isdir(FLAGS.result_dir):
    raise ValueError("Result directory does not exist!")

  pipeline_config, pipeline_config_file = _load_config(
    FLAGS.pipeline_config_file, FLAGS.result_dir, FLAGS.resume)

  if FLAGS.prefix:
    FLAGS.prefix += '_'
  if not FLAGS.resume:
    # Create the new result folder.
    result_folder_name = FLAGS.prefix + 'train_{}_{}'.format(
      os.path.basename(pipeline_config.dataset.dataset_path),
      datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    FLAGS.result_dir = os.path.join(FLAGS.result_dir, result_folder_name)
    os.mkdir(FLAGS.result_dir)

  util_ops.init_logger(FLAGS.result_dir, FLAGS.resume)

  np.random.seed(pipeline_config.seed)
  tf.set_random_seed(pipeline_config.seed)

  if FLAGS.prefix:
    FLAGS.prefix += '_'

  logging.info("Command line arguments: {}".format(sys.argv))

  def input_fn():
    return _prepare_dataset(FLAGS.result_dir,
                             pipeline_config_file,
                             pipeline_config,
                             FLAGS.prefix,
                             FLAGS.use_existing_tfrecords,
                             FLAGS.use_tfrecords)

  trainer.train(input_fn=input_fn,
                pipeline_config=pipeline_config,
                result_dir=FLAGS.result_dir,
                resume=FLAGS.resume)

if __name__ == '__main__':
  tf.app.run()
