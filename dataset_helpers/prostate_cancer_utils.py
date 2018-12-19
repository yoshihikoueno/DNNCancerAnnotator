import os
import random
import functools
import logging
import pickle
import itertools
import multiprocessing

import numpy as np
import tensorflow as tf

from utils import util_ops
from utils import standard_fields
from dataset_helpers import helpers as dh


def _get_smallest_patient_data_key(data):
  res_key = ''
  smallest_size = 99999999999

  for patient_id, patient_data in data:
    if len(patient_data) < smallest_size:
      smallest_size = len(patient_data)
      res_key = patient_id

  return res_key


def _extract_annotation(decoded_annotation, dilate_groundtruth,
                        dilate_kernel_size):
  bool_mask = tf.greater(tf.subtract(decoded_annotation[:, :, 0],
                                     decoded_annotation[:, :, 1]), 200)

  assert(len(bool_mask.get_shape().as_list()) == 2)

  annotation_mask = tf.expand_dims(tf.cast(bool_mask, dtype=tf.int32), 2)
  if dilate_groundtruth:
    annotation_mask = tf.squeeze(tf.keras.layers.MaxPool2D(
      (dilate_kernel_size, dilate_kernel_size), strides=1, padding='same',
      data_format='channels_last')(tf.expand_dims(annotation_mask, axis=0)),
                                 axis=0)

  return annotation_mask


def _preprocess_image(image_decoded, target_dims, is_annotation_mask):
  # Image should be quadratic
  image_shape = tf.shape(image_decoded)
  quadratic_assert_op = tf.Assert(tf.equal(image_shape[0], image_shape[1]),
                                  [tf.constant('Image should be quadratic.')])
  # Since we want to resize to a common size, we do it to the largest naturally
  # occuring, which should be 512x512
  size_assert_op = tf.Assert(tf.reduce_all(tf.less_equal(
    image_shape[:2], tf.constant([512, 512]))), [
      tf.constant('Largest natural image size should be <= 512')])

  with tf.control_dependencies([quadratic_assert_op, size_assert_op]):
    # Resize to common size
    # Since the prostate area is around 1/3 of the image size, we want to make
    # Sure that our crop also captures this area. However, we need to be
    # careful as unet crops off the outer parts of the image, therefore factor
    # of 2
    common_size = [max(512, int(target_dims[0] * 2)),
                   max(512, int(target_dims[1] * 2))]
    image_resized = tf.squeeze(tf.clip_by_value(tf.image.resize_area(
      tf.expand_dims(image_decoded, axis=0),
      tf.constant(common_size)), 0, 255), axis=0)

    # Get crop offset
    offset_height = int(np.floor((common_size[0] - target_dims[0]) / 2.0))
    offset_width = int(np.ceil((common_size[1] - target_dims[1]) / 2.0))

    if is_annotation_mask:
      # Make sure values are 0 or 1
      image_resized = tf.cast(tf.greater(image_resized, 0.2), tf.float32)
      # Get bounding box around masked region, in order to check if our crop
      # will cut it off
      mask_min = tf.reduce_min(tf.where(tf.equal(image_resized, 1)),
                               axis=0)[:2]
      mask_max = tf.reduce_max(tf.where(tf.equal(image_resized, 1)),
                               axis=0)[:2]

      mask_min_crop_assert = tf.Assert(
        tf.reduce_all(tf.greater(mask_min, [offset_height, offset_width])),
        data=[mask_min, [tf.constant('Groundtruth mask is cropped.')]])
      mask_max_crop_assert = tf.Assert(
        tf.reduce_all(tf.less(mask_max, [offset_height + target_dims[0],
                                           offset_width + target_dims[1]])),
        data=[mask_max, [tf.constant('Groundtruth mask is cropped.')]])
      with tf.control_dependencies([mask_min_crop_assert,
                                    mask_max_crop_assert]):
        image_cropped = tf.image.crop_to_bounding_box(
          image_resized, offset_height, offset_width, target_dims[0],
          target_dims[1])
    else:
      image_cropped = tf.image.crop_to_bounding_box(
        image_resized, offset_height, offset_width, target_dims[0],
        target_dims[1])

    return image_cropped


def _decode_example(example_dict, target_dims, dilate_groundtruth,
                    dilate_kernel_size):
  image_string = example_dict[standard_fields.TfExampleFields.image_encoded]
  image_decoded = tf.to_float(tf.image.decode_jpeg(
    image_string, channels=target_dims[2]))

  label = example_dict[standard_fields.TfExampleFields.label]

  annotation_string = example_dict[
      standard_fields.TfExampleFields.annotation_encoded]
  annotation_decoded = tf.to_float(tf.image.decode_jpeg(
      annotation_string, channels=3))

  annotation_mask = _extract_annotation(annotation_decoded, dilate_groundtruth,
                                        dilate_kernel_size)
  annotation_mask_preprocessed = _preprocess_image(
    annotation_mask, target_dims, is_annotation_mask=True)
  annotation_preprocessed = _preprocess_image(
    annotation_decoded, target_dims, is_annotation_mask=False)
  image_preprocessed = _preprocess_image(image_decoded, target_dims,
                                         is_annotation_mask=False)

  features = {
    standard_fields.InputDataFields.patient_id:
    example_dict[standard_fields.TfExampleFields.patient_id],
    standard_fields.InputDataFields.slice_id:
    example_dict[standard_fields.TfExampleFields.slice_id],
    standard_fields.InputDataFields.image_file:
    example_dict[standard_fields.TfExampleFields.image_file],
    standard_fields.InputDataFields.image_decoded: image_preprocessed,
    standard_fields.InputDataFields.annotation_file:
    example_dict[standard_fields.TfExampleFields.annotation_file],
    standard_fields.InputDataFields.annotation_decoded:
    annotation_preprocessed,
    standard_fields.InputDataFields.annotation_mask:
    annotation_mask_preprocessed,
    standard_fields.InputDataFields.label: label}

  return features


# Copies part of the data dict consisting of the keys ids into a new dict
def _make_dataset_split(data, ids):
  return dict((k, data[k]) for k in ids)


def _make_dataset_splits(data):
  patient_ids = list(data.keys())
  np.random.shuffle(patient_ids)

  train_nb = int(np.floor(len(patient_ids) * 0.8))
  val_nb = int(np.floor(len(patient_ids) * 0.15))

  train_ids = patient_ids[:train_nb]
  val_ids = patient_ids[train_nb:train_nb + val_nb]
  test_ids = patient_ids[train_nb + val_nb:]

  train_split = _make_dataset_split(data, train_ids)
  val_split = _make_dataset_split(data, val_ids)
  test_split = _make_dataset_split(data, test_ids)

  return train_split, val_split, test_split


def _sort_files(dataset_folder, balance_classes, balance_remove_smallest,
                balance_remove_random, only_cancer):
  healthy_cases_folder = os.path.join(dataset_folder, 'healthy_cases')
  cancer_cases_folder = os.path.join(dataset_folder, 'cancer_cases')
  cancer_annotations_folder = os.path.join(dataset_folder,
                                           'cancer_annotations')

  # Healthy Cases
  healthy_images = dict()
  healthy_nb = 0
  healthy_patient_nb = 0
  if not only_cancer:
    for (dirpath, _, filenames) in os.walk(healthy_cases_folder):
      if not filenames:
        continue

      patient_id = 'h' + os.path.basename(dirpath)
      assert(patient_id not in healthy_images)
      healthy_images[patient_id] = []
      healthy_patient_nb += 1
      for filename in filenames:
        healthy_images[patient_id].append([
          os.path.join(dirpath, filename), os.path.join(dirpath, filename), 0,
          patient_id, int(filename.split('.')[0])])
        healthy_nb += 1

  logging.info("Healthy Images: {}".format(healthy_nb))

  # Cancer Cases
  cancer_images = dict()
  cancer_nb = 0
  cancer_patient_nb = 0
  for (dirpath, _, filenames) in os.walk(cancer_cases_folder):
    if not filenames:
      continue
    patient_id = 'c' + os.path.basename(dirpath)
    if (patient_id in cancer_images or patient_id in healthy_images):
      logging.error("Patient ID {} is already loaded!".format(patient_id))
      assert(patient_id not in cancer_images and patient_id not in
             healthy_images)
    cancer_images[patient_id] = []
    cancer_patient_nb += 1
    for filename in filenames:
      annotation_file = os.path.join(cancer_annotations_folder,
                                     os.path.split(dirpath)[1],
                                     os.path.splitext(filename)[0]) + '.png'
      if not os.path.exists(annotation_file):
        logging.error("{} has no annotation file {}.".format(
          os.path.join(dirpath, filename), annotation_file))
        assert os.path.exists(annotation_file)
      cancer_images[patient_id].append([os.path.join(dirpath, filename),
                                        annotation_file, 1, patient_id,
                                        int(filename.split('.')[0])])
      cancer_nb += 1

  logging.info("Cancer Images: {}".format(cancer_nb))

  if balance_classes:
    assert(not only_cancer)
    logging.info("Healthy:Cancer patients {}:{}".format(healthy_patient_nb,
                                                        cancer_patient_nb))
    # Balance class distribution
    while (healthy_patient_nb != cancer_patient_nb):
      if healthy_patient_nb > cancer_patient_nb:
        if balance_remove_smallest:
          assert(not balance_remove_random)
          key = _get_smallest_patient_data_key(healthy_images)
        else:
          assert(balance_remove_random)
          key = random.choice(list(healthy_images.keys()))
        del healthy_images[key]
        healthy_patient_nb -= 1
      else:
        if balance_remove_smallest:
          assert(not balance_remove_random)
          key = _get_smallest_patient_data_key(cancer_images)
        else:
          assert(balance_remove_random)
          key = random.choice(list(cancer_images.keys()))
        del cancer_images[key]
        cancer_patient_nb -= 1

  logging.info("Final Healthy:Cancer patients {}:{}".format(healthy_patient_nb,
                                                            cancer_patient_nb))

  # Since we only train with one patient slice per epoch, we do not need to
  # Consider the actual number of images
  if healthy_patient_nb != 0:
    patient_ratio = healthy_patient_nb / float(cancer_patient_nb)
  else:
    patient_ratio = 1

  healthy_train, healthy_val, healthy_test = _make_dataset_splits(
    healthy_images)
  cancer_train, cancer_val, cancer_test = _make_dataset_splits(
    cancer_images)

  logging.info("Healthy Patient Train/Val/Test: {}/{}/{}".format(
    len(healthy_train), len(healthy_val), len(healthy_test)))
  logging.info("Cancer Patient Train/Val/Test: {}/{}/{}".format(
    len(cancer_train), len(cancer_val), len(cancer_test)))

  train_data_dict = {**healthy_train, **cancer_train}

  train_patient_ids = list(train_data_dict.keys())
  train_files = []
  train_size = 0
  for _, entries in train_data_dict.items():
    train_size += len(entries)
    for entry in entries:
      train_files.append(entry[0])

  assert(len(train_files) == train_size)

  val_data_dict = {**healthy_val, **cancer_val}

  val_patient_ids = list(val_data_dict.keys())
  val_files = []
  val_size = 0
  for _, entries in val_data_dict.items():
    val_size += len(entries)
    for entry in entries:
      val_files.append(entry[0])

  assert(len(val_files) == val_size)

  test_data_dict = {**healthy_test, **cancer_test}

  test_patient_ids = list(test_data_dict.keys())
  test_files = []
  test_size = 0
  for _, entries in test_data_dict.items():
    test_size += len(entries)
    for entry in entries:
      test_files.append(entry[0])

  assert(len(test_files) == test_size)

  dataset_size = train_size + val_size + test_size

  assert(dataset_size == cancer_nb + healthy_nb)

  logging.info("Total dataset size: {}".format(dataset_size))

  logging.info("Total Train/Val/Test Data: {}/{}/{}".format(
    train_size, val_size, test_size))

  data_dict = dict()
  data_dict[standard_fields.SplitNames.train] = train_data_dict
  data_dict[standard_fields.SplitNames.val] = val_data_dict
  data_dict[standard_fields.SplitNames.test] = test_data_dict

  return data_dict, {
    standard_fields.PickledDatasetInfo.patient_ids:
    {standard_fields.SplitNames.train: train_patient_ids,
     standard_fields.SplitNames.val: val_patient_ids,
     standard_fields.SplitNames.test: test_patient_ids},
    standard_fields.PickledDatasetInfo.file_names:
    {standard_fields.SplitNames.train: train_files,
     standard_fields.SplitNames.val: val_files,
     standard_fields.SplitNames.test: test_files},
    standard_fields.PickledDatasetInfo.patient_ratio: patient_ratio}


def _load_from_files(dataset_path, dataset_type, balance_classes,
                     balance_remove_smallest, balance_remove_random,
                     only_cancer_images,
                     input_image_dims, seed):
  assert(dataset_type == 'prostate_cancer')
  assert(os.path.exists(dataset_path))
  assert(not (balance_remove_smallest and balance_remove_random))

  data_dict, dataset_files_dict = _sort_files(
    dataset_folder=dataset_path,
    balance_classes=balance_classes,
    balance_remove_smallest=balance_remove_smallest,
    balance_remove_random=balance_remove_random,
    only_cancer=only_cancer_images)

  pickle_data = {
    standard_fields.PickledDatasetInfo.seed: seed,
    standard_fields.PickledDatasetInfo.patient_ids:
    dataset_files_dict[standard_fields.PickledDatasetInfo.patient_ids],
    standard_fields.PickledDatasetInfo.file_names:
    dataset_files_dict[standard_fields.PickledDatasetInfo.file_names],
    standard_fields.PickledDatasetInfo.patient_ratio:
    dataset_files_dict[standard_fields.PickledDatasetInfo.patient_ratio]}

  return data_dict, pickle_data


def _serialize_example(example):
  patient_id = example[0]
  slice_id = example[1]
  image_file = example[2]
  image_encoded = example[3]
  annotation_file = example[4]
  annotation_encoded = example[5]
  label = example[6]

  feature = {
    standard_fields.TfExampleFields.patient_id: dh.bytes_feature(
      patient_id),
    standard_fields.TfExampleFields.slice_id: dh.int64_feature(
      slice_id),
    standard_fields.TfExampleFields.image_file: dh.bytes_feature(
      image_file),
    standard_fields.TfExampleFields.image_encoded: dh.bytes_feature(
      image_encoded),
    standard_fields.TfExampleFields.annotation_file: dh.bytes_feature(
      annotation_file),
    standard_fields.TfExampleFields.annotation_encoded: dh.bytes_feature(
      annotation_encoded),
    standard_fields.TfExampleFields.label: dh.int64_feature(
      label)}
  tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

  return tf_example.SerializeToString()


def _deserialize_and_decode_example(example, target_dims, dilate_groundtruth,
                                    dilate_kernel_size):
  features = {
    standard_fields.TfExampleFields.patient_id: tf.FixedLenFeature(
      (), tf.string, default_value=''),
    standard_fields.TfExampleFields.slice_id: tf.FixedLenFeature(
      (), tf.int64, default_value=0),
    standard_fields.TfExampleFields.image_file: tf.FixedLenFeature(
      (), tf.string, default_value=''),
    standard_fields.TfExampleFields.image_encoded: tf.FixedLenFeature(
      (), tf.string, default_value=''),
    standard_fields.TfExampleFields.annotation_file: tf.FixedLenFeature(
      (), tf.string, default_value=''),
    standard_fields.TfExampleFields.annotation_encoded:
    tf.FixedLenFeature((), tf.string, default_value=''),
    standard_fields.TfExampleFields.label: tf.FixedLenFeature(
      (), tf.int64, default_value=0)}

  example_dict = tf.parse_single_example(example, features)

  return _decode_example(example_dict, target_dims=target_dims,
                         dilate_groundtruth=dilate_groundtruth,
                         dilate_kernel_size=dilate_kernel_size)


def _parse_from_file(image_file, annotation_file, label, patient_id,
                     slice_id):
  image_string = tf.read_file(image_file)
  annotation_string = tf.read_file(annotation_file)

  return {
    standard_fields.TfExampleFields.patient_id: patient_id,
    standard_fields.TfExampleFields.slice_id: slice_id,
    standard_fields.TfExampleFields.image_file: image_file,
    standard_fields.TfExampleFields.image_encoded: image_string,
    standard_fields.TfExampleFields.annotation_file: annotation_file,
    standard_fields.TfExampleFields.annotation_encoded: annotation_string,
    standard_fields.TfExampleFields.label: label}


def build_tfrecords_from_files(
    dataset_path, dataset_type, balance_classes, balance_remove_smallest,
    balance_remove_random,
    only_cancer_images, input_image_dims, seed, output_dir):
  data_dict, pickle_data = \
    _load_from_files(
      dataset_path=dataset_path, dataset_type=dataset_type,
      balance_classes=balance_classes,
      balance_remove_smallest=balance_remove_smallest,
      balance_remove_random=balance_remove_random,
      only_cancer_images=only_cancer_images,
      input_image_dims=input_image_dims,
      seed=seed)

  logging.info("Creating patient tfrecords.")
  with tf.Session() as sess:
    os.mkdir(os.path.join(output_dir, 'tfrecords'))
    for split, data in data_dict.items():
      os.mkdir(os.path.join(output_dir, 'tfrecords', split))

      writer_dict = dict()
      # Create writers
      for patient_id in data.keys():
        writer = tf.python_io.TFRecordWriter(os.path.join(
          output_dir, 'tfrecords', split, patient_id + '.tfrecords'))
        writer_dict[patient_id] = writer

      dataset = tf.data.Dataset.from_tensor_slices(
        tuple([list(t) for t in zip(*list(
          itertools.chain.from_iterable(data.values())))]))
      dataset = dataset.map(_parse_from_file,
                            num_parallel_calls=util_ops.get_cpu_count())
      dataset = dataset.batch(512)

      it = dataset.make_one_shot_iterator()

      elem_batch = it.get_next()

      pool = multiprocessing.Pool(processes=util_ops.get_cpu_count())

      while True:
        try:
          elem_batch_result = sess.run(elem_batch)

          elem_batch_serialized = list(zip(*elem_batch_result.values()))
          elem_batch_serialized = pool.map(_serialize_example,
                                           elem_batch_serialized)

          for i, elem_serialized in enumerate(elem_batch_serialized):
            writer_dict[elem_batch_result[
              standard_fields.TfExampleFields.patient_id][i].decode(
                'utf-8')].write(elem_serialized)
        except tf.errors.OutOfRangeError:
          break

      for writer in writer_dict.values():
        writer.close()

      pool.close()

  logging.info("Finished creating patient tfrecords.")
  f = open(os.path.join(
    output_dir, standard_fields.PickledDatasetInfo.pickled_file_name), 'wb')
  pickle.dump(pickle_data, f)
  f.close()


# patient_ids are only needed for train mode
def build_tf_dataset_from_tfrecords(directory, split_name, target_dims,
                                    patient_ids, is_training,
                                    dilate_groundtruth, dilate_kernel_size):
  tfrecords_folder = os.path.join(directory, 'tfrecords', split_name)
  assert(os.path.exists(tfrecords_folder))

  tfrecords_files = os.listdir(tfrecords_folder)
  tfrecords_files = [os.path.join(tfrecords_folder, file_name)
                     for file_name in tfrecords_files]

  dataset = tf.data.Dataset.from_tensor_slices(tfrecords_files)
  if is_training:
    # We want to even out patient distribution across one epoch
    dataset = dataset.interleave(
      lambda tfrecords_file: tf.data.TFRecordDataset(tfrecords_file).apply(
        tf.data.experimental.shuffle_and_repeat(32, None)), block_length=1,
      cycle_length=len(tfrecords_files),
      num_parallel_calls=util_ops.get_cpu_count())
  else:
    dataset = dataset.interleave(
      lambda tfrecords_file: tf.data.TFRecordDataset(tfrecords_file).shuffle(
        32), block_length=1, cycle_length=len(tfrecords_files),
      num_parallel_calls=util_ops.get_cpu_count())

  deserialize_and_decode_fn = functools.partial(
    _deserialize_and_decode_example, target_dims=target_dims,
    dilate_groundtruth=dilate_groundtruth,
    dilate_kernel_size=dilate_kernel_size)

  dataset = dataset.map(deserialize_and_decode_fn,
                        num_parallel_calls=util_ops.get_cpu_count())

  return dataset
