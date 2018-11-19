import os
import pickle
from shutil import copy

from dataset_helpers import prostate_cancer_utils as pc
from utils import standard_fields


def _load_existing_tfrecords(directory, split_name, target_dims, dataset_name):
  tfrecords_file = os.path.join(directory, split_name + '.tfrecords')
  meta_file = os.path.join(
    directory, standard_fields.PickledDatasetInfo.pickled_file_name)

  f = open(meta_file, 'rb')
  meta_data = pickle.load(f)
  f.close()

  ids = meta_data[standard_fields.PickledDatasetInfo.patient_ids][split_name]

  if dataset_name == 'prostate_cancer':
    return pc.build_tf_dataset_from_tfrecords(
      tfrecords_file, target_dims, ids,
      split_name == standard_fields.SplitNames.train), meta_data
  else:
    assert(False)


def build_dataset(dataset_config, directory, existing_tfrecords_directory,
                  split_name, target_dims, seed, batch_size, shuffle,
                  shuffle_buffer_size):
  assert split_name in standard_fields.SplitNames.available_names

  dataset_name = dataset_config.WhichOneof('dataset_type')

  if existing_tfrecords_directory:
    dataset, meta_info = _load_existing_tfrecords(
      existing_tfrecords_directory, split_name, target_dims, dataset_name)

  else:
    # We need to create a tfrecords, and use it as dataset

    # Tfrecords files and meta file are created in result dir
    if dataset_name == 'prostate_cancer':
      pc.build_tfrecords_from_files(dataset_config, target_dims,
                                    seed, directory)
    else:
      assert(False)

    # Load tfrecords into dataset
    dataset, meta_info = _load_existing_tfrecords(
      directory, split_name, target_dims, dataset_name)

  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size)

  if split_name == standard_fields.SplitNames.train:
    dataset = dataset.repeat(None)

  dataset = dataset.batch(batch_size)

  # Buffer size of None means autotune
  dataset = dataset.prefetch(None)

  return dataset
