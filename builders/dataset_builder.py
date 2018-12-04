import os
import pickle

import tensorflow as tf

from dataset_helpers import prostate_cancer_utils as pc
from utils import standard_fields


def _load_existing_tfrecords(directory, split_name, target_dims, dataset_name,
                             dataset_info, is_training):
  tfrecords_file = os.path.join(directory, split_name + '.tfrecords')

  ids = dataset_info[standard_fields.PickledDatasetInfo.patient_ids][
    split_name]

  if dataset_name == 'prostate_cancer':
    return pc.build_tf_dataset_from_tfrecords(
      tfrecords_file, target_dims, patient_ids=ids, is_training=is_training)
  else:
    assert(False)


# Create tfrecords files if necessary and return meta info as dict
def prepare_dataset(dataset_config, directory, existing_tfrecords,
                    target_dims, seed):
  if existing_tfrecords:
    meta_file = os.path.join(
      directory, standard_fields.PickledDatasetInfo.pickled_file_name)

    f = open(meta_file, 'rb')
    meta_data = pickle.load(f)
    f.close()

    return meta_data
  else:
    dataset_name = dataset_config.WhichOneof('dataset_type')

    # We need to create a tfrecords, and use it as dataset

    # Tfrecords files and meta file are created in result dir
    if dataset_name == 'prostate_cancer':
      pc.build_tfrecords_from_files(dataset_config, target_dims,
                                    seed, directory)
    else:
      assert(False)

    meta_file = os.path.join(
      directory, standard_fields.PickledDatasetInfo.pickled_file_name)

    f = open(meta_file, 'rb')
    meta_data = pickle.load(f)
    f.close()

    return meta_data


def build_dataset(dataset_config, directory,
                  split_name, target_dims, seed, batch_size, shuffle,
                  shuffle_buffer_size, is_training, dataset_info):
  assert split_name in standard_fields.SplitNames.available_names

  dataset_name = dataset_config.WhichOneof('dataset_type')

  dataset = _load_existing_tfrecords(
      directory, split_name, target_dims, dataset_name,
      dataset_info, is_training=is_training)

  #if shuffle and is_training:
  #  dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
   #   shuffle_buffer_size, None))
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size)
  if is_training:
    dataset = dataset.repeat(None)


  dataset = dataset.batch(batch_size)

  # Buffer size of None means autotune
  dataset = dataset.prefetch(None)

  return dataset
