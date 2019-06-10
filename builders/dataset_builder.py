import os
import pickle
import functools

import tensorflow as tf

from dataset_helpers import prostate_cancer_utils as pc
from utils import standard_fields
from utils import preprocessor


def _load_existing_tfrecords(directory, split_name, target_dims, dataset_name,
                             dataset_info, is_training, model_objective,
                             dataset_config, target_depth):
  ids = dataset_info[standard_fields.PickledDatasetInfo.patient_ids][
    split_name]

  if dataset_name == 'prostate_cancer':
    return pc.build_tf_dataset_from_tfrecords(
      directory, split_name=split_name, target_dims=target_dims,
      patient_ids=ids, is_training=is_training,
      dilate_groundtruth=dataset_config.prostate_cancer.dilate_groundtruth,
      dilate_kernel_size=dataset_config.
      prostate_cancer.groundtruth_dilation_kernel_size,
      common_size_factor=dataset_config.prostate_cancer.common_size_factor,
      model_objective=model_objective,
      tfrecords_type=dataset_config.tfrecords_type,
      target_depth=target_depth)
  else:
    assert(False)


# Create tfrecords files if necessary and return meta info as dict
def prepare_dataset(dataset_config, directory, existing_tfrecords,
                    target_dims, seed):
  meta_file = os.path.join(dataset_config.dataset_path,
                           dataset_config.dataset_info_file)
  with open(meta_file, 'rb') as f:
    meta_data = pickle.load(f)

  dataset_name = dataset_config.WhichOneof('dataset_type')

  output_dir = os.path.join(directory, 'tfrecords')

  if not os.path.exists(output_dir) or not os.path.exists(
      os.path.join(output_dir, 'SUCCESS')):
    # Tfrecords files are created in result dir
    if dataset_name == 'prostate_cancer':
      pc.build_tfrecords_from_files(
        dataset_path=dataset_config.dataset_path,
        dataset_info_file=dataset_config.dataset_info_file,
        only_cancer_images=dataset_config.prostate_cancer.only_cancer_images,
        input_image_dims=target_dims, seed=seed, output_dir=output_dir,
        tfrecords_type=dataset_config.tfrecords_type)
    else:
      assert(False)

    # Create marker file that indicates that the tfrecords were successfully
    # created
    open(os.path.join(output_dir, 'SUCCESS'), 'a').close()

  return meta_data


def _make_gan_compatible_input(features):
  return (features[standard_fields.InputDataFields.image_decoded],
          features[standard_fields.InputDataFields.annotation_mask])


def build_dataset(dataset_name, directory,
                  split_name, target_dims, seed, batch_size, shuffle,
                  shuffle_buffer_size, is_training, dataset_info,
                  dataset_config, is_gan_model, data_augmentation_options,
                  num_parallel_iterations, model_objective,
                  target_depth):
  assert split_name in standard_fields.SplitNames.available_names

  dataset = _load_existing_tfrecords(
    directory, split_name, target_dims, dataset_name,
    dataset_info, is_training=is_training, dataset_config=dataset_config,
    model_objective=model_objective, target_depth=target_depth)

  if shuffle and is_training:
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
      shuffle_buffer_size, None))
  elif is_training:
    dataset = dataset.repeat(None)

  if is_training:
    # Since we have repeat(None),
    # nothing will be dropped anyway, and now we have a defined batch dim
    dataset = dataset.batch(batch_size, drop_remainder=True)
  else:
    assert(batch_size == 1)
    # We need a defined shape, once we set batch size > 1 we cannot drop
    # remainder anymore
    dataset = dataset.batch(batch_size, drop_remainder=True)

  # Buffer size of None means autotune
  dataset = dataset.prefetch(None)

  if is_training:
    # Apply data augmentation
    augment_fn = functools.partial(
      preprocessor.apply_data_augmentation,
      data_augmentation_options=data_augmentation_options,
      num_parallel_iterations=num_parallel_iterations)
    dataset = dataset.map(
      augment_fn, num_parallel_calls=num_parallel_iterations)

  # General Preprocessing
  preprocess_fn = functools.partial(
    preprocessor.preprocess, val_range=dataset_config.val_range,
    scale_input=dataset_config.scale_input, model_objective=model_objective,
    tfrecords_type=dataset_config.tfrecords_type)
  dataset = dataset.map(
    preprocess_fn, num_parallel_calls=num_parallel_iterations)

  if is_gan_model:
    # The current GANEstimator expects inputs of the shape (features, labels),
    # Where features is the generator input, and labels the potential input
    # for the discriminator
    dataset = dataset.map(_make_gan_compatible_input,
                          num_parallel_calls=num_parallel_iterations)

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
