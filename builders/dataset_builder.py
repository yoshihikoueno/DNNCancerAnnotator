import os
import pickle

import tensorflow as tf

from dataset_helpers import prostate_cancer_utils as pc
from utils import standard_fields


def _load_existing_tfrecords(directory, split_name, target_dims, dataset_name,
                             dataset_info, is_training, dataset_config):
  ids = dataset_info[standard_fields.PickledDatasetInfo.patient_ids][
    split_name]

  if dataset_name == 'prostate_cancer':
    return pc.build_tf_dataset_from_tfrecords(
      directory, split_name=split_name, target_dims=target_dims,
      patient_ids=ids, is_training=is_training,
      dilate_groundtruth=dataset_config.prostate_cancer.dilate_groundtruth,
      dilate_kernel_size=dataset_config.
      prostate_cancer.groundtruth_dilation_kernel_size,
      common_size_factor=dataset_config.prostate_cancer.common_size_factor)
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
      dataset_type = dataset_config.WhichOneof('dataset_type')
      balance_remove_smallest = (dataset_config.prostate_cancer.
                                 balance_remove_smallest_patient_set)
      balance_remove_random = (dataset_config.prostate_cancer.
                               balance_remove_random_patient_set)
      pc.build_tfrecords_from_files(
        dataset_path=dataset_config.dataset_path, dataset_type=dataset_type,
        balance_classes=dataset_config.balance_classes,
        balance_remove_smallest=balance_remove_smallest,
        balance_remove_random=balance_remove_random,
        only_cancer_images=dataset_config.prostate_cancer.only_cancer_images,
        input_image_dims=target_dims, seed=seed, output_dir=directory)
    else:
      assert(False)

    meta_file = os.path.join(
      directory, standard_fields.PickledDatasetInfo.pickled_file_name)

    f = open(meta_file, 'rb')
    meta_data = pickle.load(f)
    f.close()

    return meta_data


def build_dataset(dataset_name, directory,
                  split_name, target_dims, seed, batch_size, shuffle,
                  shuffle_buffer_size, is_training, dataset_info,
                  dataset_config):
  assert split_name in standard_fields.SplitNames.available_names

  dataset = _load_existing_tfrecords(
      directory, split_name, target_dims, dataset_name,
      dataset_info, is_training=is_training, dataset_config=dataset_config)

  if shuffle and is_training:
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
      shuffle_buffer_size, None))
  elif shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size)
  elif is_training:
    dataset = dataset.repeat(None)

  if is_training:
    # Since we have repeat(None),
    # nothing will be dropped anyway, and now we have a defined batch dim
    dataset = dataset.batch(batch_size, drop_remainder=True)
  else:
    dataset = dataset.batch(batch_size, drop_remainder=False)

  # Buffer size of None means autotune
  dataset = dataset.prefetch(None)

  return dataset


def build_predict_dataset(dataset_name, input_dir, target_dims,
                          dataset_config):
  if dataset_name == 'prostate_cancer':
    dataset = pc.build_predict_tf_dataset(
      directory=input_dir, target_dims=target_dims,
      common_size_factor=dataset_config.prostate_cancer.common_size_factor)
  else:
    assert(False)

  dataset = dataset.batch(1)

  return dataset
