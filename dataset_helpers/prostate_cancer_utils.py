import os
import functools
import logging
import pickle
import itertools
import shutil

import numpy as np
import tensorflow as tf

from utils import util_ops
from utils import standard_fields
from dataset_helpers import helpers as dh


# mask = [HxW]
def split_mask(mask, dilate_mask=False):
  assert(len(mask.get_shape()) == 2)
  assert(mask.dtype == tf.int64)

  if dilate_mask:
    mask = tf.squeeze(tf.keras.layers.MaxPool2D(
      (5, 5), strides=1, padding='same',
      data_format='channels_last')(
        tf.expand_dims(tf.expand_dims(mask, axis=0), axis=3)))

  # Label each area with individual index
  components = tf.contrib.image.connected_components(mask)

  if dilate_mask:
    # we need to erode the mask again
    components = tf.where(tf.equal(mask, tf.constant(0, dtype=tf.int64)),
                          tf.zeros_like(components, dtype=tf.int32),
                          components)

  unique_ids, unique_indices = tf.unique(tf.reshape(components, [-1]))

  # Remove zero id, since it describes background
  unique_ids = tf.gather_nd(unique_ids, tf.where(tf.not_equal(unique_ids, 0)))

  # Create mask for each cancer area
  individual_masks = tf.map_fn(
    lambda unique_id: tf.equal(unique_id, components), elems=unique_ids,
    dtype=tf.bool, parallel_iterations=4)

  return individual_masks

# def _extract_bounding_box(groundtruth_mask):
#   assert(len(groundtruth_mask.get_shape()) == 2)
#   assert(groundtruth_mask.dtype == tf.bool)

#   indices = tf.transpose(tf.where(groundtruth_mask))

#   groundtruth_shape = groundtruth_mask.get_shape().as_list()
#   y_min = tf.div(tf.to_float(tf.reduce_min(indices[0])),
#                  tf.constant(groundtruth_shape[0], dtype=tf.float32))
#   y_max = tf.div(tf.to_float(tf.reduce_min(indices[0])),
#                  tf.constant(groundtruth_shape[0], dtype=tf.float32))
#   x_min = tf.div(tf.to_float(tf.reduce_min(indices[1])),
#                  tf.constant(groundtruth_shape[1], dtype=tf.float32))
#   x_max = tf.div(tf.to_float(tf.reduce_min(indices[1])),
#                  tf.constant(groundtruth_shape[1], dtype=tf.float32))

#   return [y_min, y_max, x_min, x_max]


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


def _preprocess_image(image_decoded, target_dims, is_annotation_mask,
                      common_size_factor):
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
    common_size = [max(512, int(target_dims[0] * common_size_factor)),
                   max(512, int(target_dims[1] * common_size_factor))]
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
                    dilate_kernel_size, common_size_factor):
  image_string = example_dict[standard_fields.TfExampleFields.image_encoded]
  image_decoded = tf.cast(tf.image.decode_jpeg(
    image_string, channels=target_dims[2]), tf.float32)

  label = example_dict[standard_fields.TfExampleFields.label]

  annotation_string = example_dict[
      standard_fields.TfExampleFields.annotation_encoded]
  annotation_decoded = tf.cast(tf.image.decode_jpeg(
      annotation_string, channels=3), tf.float32)

  same_size_assert = tf.Assert(
    tf.reduce_all(tf.equal(tf.shape(annotation_decoded)[:2],
                           tf.shape(image_decoded)[:2])),
    data=[tf.shape(annotation_decoded)[:2],
          tf.shape(image_decoded)[:2],
          example_dict[standard_fields.TfExampleFields.image_file],
          [tf.constant(
            'Annotation and original image not same size.')]])

  with tf.control_dependencies([same_size_assert]):
    annotation_mask = _extract_annotation(
      annotation_decoded, dilate_groundtruth, dilate_kernel_size)

  annotation_mask_preprocessed = _preprocess_image(
    annotation_mask, target_dims, is_annotation_mask=True,
    common_size_factor=common_size_factor)
  annotation_preprocessed = _preprocess_image(
    annotation_decoded, target_dims, is_annotation_mask=False,
    common_size_factor=common_size_factor)
  image_preprocessed = _preprocess_image(
    image_decoded, target_dims, is_annotation_mask=False,
    common_size_factor=common_size_factor)

  #individual_masks = split_groundtruth_mask(tf.squeeze(tf.cast(
  #  annotation_mask_preprocessed, tf.bool), axis=2))

  #bounding_box_coordinates = tf.cond(
  #  tf.equal(label, 0), lambda: [tf.constant([], dtype=tf.float32),
  #                               tf.constant([], dtype=tf.float32),
  #                               tf.constant([], dtype=tf.float32),
  #                               tf.constant([], dtype=tf.float32)],
  #  lambda: tf.map_fn(_extract_bounding_box,
  #                    elems=individual_masks,
  #                    dtype=[tf.float32, tf.float32, tf.float32, tf.float32]))

  #y_mins = bounding_box_coordinates[0]
  #y_maxs = bounding_box_coordinates[1]
  #x_mins = bounding_box_coordinates[2]
  #x_maxs = bounding_box_coordinates[3]

  #bounding_boxes = {standard_fields.BoundingBoxFields.y_min: y_mins,
  #                  standard_fields.BoundingBoxFields.y_max: y_maxs,
  #                  standard_fields.BoundingBoxFields.x_min: x_mins,
  #                  standard_fields.BoundingBoxFields.x_max: x_maxs}

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
    #standard_fields.InputDataFields.individual_masks:
    #individual_masks,
    #standard_fields.InputDataFields.bounding_boxes:
    #bounding_boxes,
    standard_fields.InputDataFields.label: label,
    standard_fields.InputDataFields.examination_name: example_dict[
      standard_fields.TfExampleFields.examination_name]}

  return features


def _serialize_example(keys, example):
  patient_id = example[keys.index(standard_fields.TfExampleFields.patient_id)]
  slice_id = example[keys.index(standard_fields.TfExampleFields.slice_id)]
  image_file = example[keys.index(standard_fields.TfExampleFields.image_file)]
  image_encoded = example[keys.index(
    standard_fields.TfExampleFields.image_encoded)]
  annotation_file = example[keys.index(
    standard_fields.TfExampleFields.annotation_file)]
  annotation_encoded = example[keys.index(
    standard_fields.TfExampleFields.annotation_encoded)]
  label = example[keys.index(standard_fields.TfExampleFields.label)]
  examination_name = example[keys.index(
    standard_fields.TfExampleFields.examination_name)]

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
      label),
    standard_fields.TfExampleFields.examination_name: dh.bytes_feature(
      examination_name)}
  tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

  return tf_example.SerializeToString()


def _deserialize_and_decode_example(example, target_dims, dilate_groundtruth,
                                    dilate_kernel_size, common_size_factor):
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
      (), tf.int64, default_value=0),
    standard_fields.TfExampleFields.examination_name: tf.FixedLenFeature(
      (), tf.string, default_value='')}

  example_dict = tf.parse_single_example(example, features)

  return _decode_example(example_dict, target_dims=target_dims,
                         dilate_groundtruth=dilate_groundtruth,
                         dilate_kernel_size=dilate_kernel_size,
                         common_size_factor=common_size_factor)


def _parse_from_file(image_file, annotation_file, label, patient_id,
                     slice_id, examination_name, dataset_folder):
  image_file = tf.strings.join([dataset_folder, '/', image_file])
  image_string = tf.read_file(image_file)

  annotation_file = tf.strings.join([dataset_folder, '/', annotation_file])
  annotation_string = tf.read_file(annotation_file)

  return {
    standard_fields.TfExampleFields.patient_id: patient_id,
    standard_fields.TfExampleFields.slice_id: slice_id,
    standard_fields.TfExampleFields.image_file: image_file,
    standard_fields.TfExampleFields.image_encoded: image_string,
    standard_fields.TfExampleFields.annotation_file: annotation_file,
    standard_fields.TfExampleFields.annotation_encoded: annotation_string,
    standard_fields.TfExampleFields.label: label,
    standard_fields.TfExampleFields.examination_name: examination_name}


def build_tfrecords_from_files(
    dataset_path, dataset_info_file,
    only_cancer_images, input_image_dims, seed, output_dir):
  if os.path.exists(output_dir):
    # Clean everything inside
    shutil.rmtree(output_dir)

  os.mkdir(output_dir)

  dataset_info_file = os.path.join(dataset_path, dataset_info_file)
  if not os.path.exists(dataset_info_file):
    raise ValueError("Pickled dataset info file missing!")

  with open(dataset_info_file, 'rb') as f:
    pickle_data = pickle.load(f)

  logging.info("Creating patient tfrecords.")
  with tf.Session() as sess:
    for split, data in pickle_data[
        standard_fields.PickledDatasetInfo.data_dict].items():
      os.mkdir(os.path.join(output_dir, split))

      writer_dict = dict()
      # Create writers
      for patient_id in data.keys():
        writer = tf.python_io.TFRecordWriter(os.path.join(
          output_dir, split, patient_id + '.tfrecords'))
        writer_dict[patient_id] = writer

      dataset = tf.data.Dataset.from_tensor_slices(
        tuple([list(t) for t in zip(*list(
          itertools.chain.from_iterable(data.values())))]))

      parse_fn = functools.partial(_parse_from_file,
                                   dataset_folder=dataset_path)
      dataset = dataset.map(parse_fn,
                            num_parallel_calls=util_ops.get_cpu_count())
      dataset = dataset.batch(40)

      it = dataset.make_one_shot_iterator()

      elem_batch = it.get_next()

      while True:
        try:
          elem_batch_result = sess.run(elem_batch)
          keys = list(elem_batch_result.keys())
          elem_batch_serialized = list(zip(*elem_batch_result.values()))

          # Unfortunately for some reason we cannot use multiprocessing here.
          # Sometimes the map call will freeze
          elem_batch_serialized = list(map(
            lambda v: _serialize_example(keys, v),
            elem_batch_serialized))

          for i, elem_serialized in enumerate(elem_batch_serialized):
            writer_dict[elem_batch_result[
              standard_fields.TfExampleFields.patient_id][i].decode(
                'utf-8')].write(elem_serialized)
        except tf.errors.OutOfRangeError:
          break

      for writer in writer_dict.values():
        writer.close()

  logging.info("Finished creating patient tfrecords.")


# patient_ids are only needed for train mode
def build_tf_dataset_from_tfrecords(directory, split_name, target_dims,
                                    patient_ids, is_training,
                                    dilate_groundtruth, dilate_kernel_size,
                                    common_size_factor):
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
    dilate_kernel_size=dilate_kernel_size,
    common_size_factor=common_size_factor)

  dataset = dataset.map(deserialize_and_decode_fn,
                        num_parallel_calls=util_ops.get_cpu_count())

  return dataset


def build_predict_tf_dataset(directory, target_dims, common_size_factor):
  files = os.listdir(directory)
  for i, f in enumerate(files):
    files[i] = os.path.join(directory, f)
  files = [f for f in files if os.path.isfile(f)]

  dataset = tf.data.Dataset.from_tensor_slices(files)

  preprocess_fn = functools.partial(
    _prepare_predict_example, target_dims=target_dims,
    common_size_factor=common_size_factor)

  dataset = dataset.map(
    preprocess_fn, num_parallel_calls=util_ops.get_cpu_count())

  return dataset


def _prepare_predict_example(image_file, target_dims, common_size_factor):
  image_string = tf.read_file(image_file)

  image_decoded = tf.cast(tf.image.decode_jpeg(
    image_string, channels=target_dims[2]), tf.float32)

  image_preprocessed = _preprocess_image(
    image_decoded, target_dims=target_dims, is_annotation_mask=False,
    common_size_factor=common_size_factor)

  return {standard_fields.InputDataFields.image_decoded: image_preprocessed,
          standard_fields.InputDataFields.image_file: image_file}
