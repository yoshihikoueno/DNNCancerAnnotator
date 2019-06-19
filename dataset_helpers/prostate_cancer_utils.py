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


def split_mask(mask, dilate_mask=False, is_3d=False):
  assert(mask.dtype == tf.int64)
  if is_3d:
    assert(len(mask.get_shape()) == 3)
    pool_op = tf.keras.layers.MaxPool3D
  else:
    assert(len(mask.get_shape()) == 2)
    pool_op = tf.keras.layers.MaxPool2D

  if dilate_mask:
    if is_3d:
      # Unfortunately 3D Pooling does not support int64
      mask = tf.squeeze(tf.cast(pool_op(
        5, strides=1, padding='same',
        data_format='channels_last')(
          tf.expand_dims(tf.cast(tf.expand_dims(mask, axis=0), tf.float32),
                         axis=-1)), tf.int64))
    else:
      mask = tf.squeeze(pool_op(
        5, strides=1, padding='same',
        data_format='channels_last')(
          tf.expand_dims(tf.expand_dims(mask, axis=0), axis=-1)))

  # Label each area with individual index
  components = tf.contrib.image.connected_components(mask)

  if dilate_mask:
    # we need to shrink the mask again by masking with original
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
    standard_fields.InputDataFields.label: label,
    standard_fields.InputDataFields.examination_name: example_dict[
      standard_fields.TfExampleFields.examination_name]}

  return features


def _decode_3d_example(example_dict, target_dims, dilate_groundtruth,
                       dilate_kernel_size, common_size_factor, target_depth,
                       is_training):
  context_features, sequence_features = example_dict

  patient_id = context_features[standard_fields.TfExampleFields.patient_id]
  examination_name = context_features[
    standard_fields.TfExampleFields.examination_name]

  image_3d_encoded = sequence_features[
    standard_fields.TfExampleFields.image_3d_encoded].values
  annotation_3d_encoded = sequence_features[
    standard_fields.TfExampleFields.annotation_3d_encoded].values

  image_decoded = tf.cast(tf.map_fn(
    lambda e: tf.image.decode_jpeg(e, channels=1),
    elems=image_3d_encoded,
    parallel_iterations=util_ops.get_cpu_count(), dtype=tf.uint8), tf.float32)
  annotation_decoded = tf.cast(tf.map_fn(
    lambda e: tf.image.decode_jpeg(e, channels=3),
    elems=annotation_3d_encoded,
    parallel_iterations=util_ops.get_cpu_count(), dtype=tf.uint8), tf.float32)

  annotation_mask = tf.map_fn(
    lambda e: _extract_annotation(e, dilate_groundtruth=dilate_groundtruth,
                                  dilate_kernel_size=dilate_kernel_size),
    parallel_iterations=util_ops.get_cpu_count(),
    elems=annotation_decoded, dtype=tf.int32)

  image_files = sequence_features[
    standard_fields.TfExampleFields.image_file].values
  annotation_files = sequence_features[
    standard_fields.TfExampleFields.annotation_file].values
  slice_ids = sequence_features[
        standard_fields.TfExampleFields.slice_id].values

  if is_training:
    # Modify depth dimension
    total_depth = tf.shape(image_decoded)[0]
    depth_assert = tf.Assert(
      tf.greater_equal(total_depth, target_depth),
      data=[tf.shape(image_decoded), total_depth, tf.constant(target_depth),
            tf.constant("Not enough image slices.")])
    with tf.control_dependencies([depth_assert]):
      first_slice_index = tf.random.uniform(
        shape=[], minval=0, maxval=total_depth - tf.constant(target_depth - 1),
        dtype=tf.int32)
      image_decoded = image_decoded[
        first_slice_index:first_slice_index + target_depth]
      annotation_decoded = annotation_decoded[
        first_slice_index:first_slice_index + target_depth]
      annotation_mask = annotation_mask[
        first_slice_index:first_slice_index + target_depth]
      image_files = image_files[
        first_slice_index:first_slice_index + target_depth]
      annotation_files = annotation_files[
        first_slice_index:first_slice_index + target_depth]
      slice_ids = slice_ids[first_slice_index:first_slice_index + target_depth]

      shape = image_decoded.get_shape()
      image_decoded.set_shape([target_depth, shape[1], shape[2], shape[3]])

      shape = annotation_decoded.get_shape()
      annotation_decoded.set_shape([target_depth, shape[1], shape[2], shape[3]])

      shape = annotation_mask.get_shape()
      annotation_mask.set_shape([target_depth, shape[1], shape[2], shape[3]])

  image_preprocessed = tf.map_fn(lambda s: _preprocess_image(
    image_decoded=s, target_dims=target_dims,
    is_annotation_mask=False,
    common_size_factor=common_size_factor), elems=image_decoded,
                                 dtype=tf.float32)
  annotation_preprocessed = tf.map_fn(lambda s: _preprocess_image(
    image_decoded=s, target_dims=target_dims,
    is_annotation_mask=False,
    common_size_factor=common_size_factor), elems=annotation_decoded,
                                 dtype=tf.float32)
  annotation_mask_preprocessed = tf.map_fn(lambda s: _preprocess_image(
    image_decoded=s, target_dims=target_dims,
    is_annotation_mask=True,
    common_size_factor=common_size_factor), elems=annotation_mask,
                                           dtype=tf.float32)

  features = {
    standard_fields.InputDataFields.patient_id: patient_id,
    standard_fields.InputDataFields.image_decoded: image_preprocessed,
    standard_fields.InputDataFields.annotation_decoded:
    annotation_preprocessed,
    standard_fields.InputDataFields.annotation_mask:
    annotation_mask_preprocessed,
    standard_fields.InputDataFields.examination_name: examination_name,
    standard_fields.InputDataFields.image_file: image_files,
    standard_fields.InputDataFields.annotation_file: annotation_files,
    standard_fields.InputDataFields.slice_id: slice_ids}

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


def _serialize_3d_example(example_data):
  default_patient_id = example_data[2][0]
  default_exam_id = example_data[4][0]

  # Verify data
  slice_index_counter = None

  for image_file, annotation_file, patient_id, slice_index, exam_id, \
       image_decoded, annotation_decoded in zip(*example_data):
    assert(default_patient_id == patient_id)
    assert(default_exam_id == exam_id)
    if slice_index_counter is None:
      slice_index_counter = slice_index
    else:
      assert(slice_index_counter + 1 == slice_index)
      slice_index_counter = slice_index

  context_features = {
    standard_fields.TfExampleFields.patient_id: dh.bytes_feature(
      default_patient_id),
    standard_fields.TfExampleFields.examination_name: dh.bytes_feature(
      default_exam_id)
  }
  context_features = tf.train.Features(feature=context_features)

  feature_lists = {
    standard_fields.TfExampleFields.image_3d_encoded:
    tf.train.FeatureList(feature=[dh.bytes_list_feature(example_data[5])]),
    standard_fields.TfExampleFields.image_file:
    tf.train.FeatureList(feature=[dh.bytes_list_feature(example_data[0])]),
    standard_fields.TfExampleFields.annotation_3d_encoded:
    tf.train.FeatureList(feature=[dh.bytes_list_feature(example_data[6])]),
    standard_fields.TfExampleFields.annotation_file:
    tf.train.FeatureList(feature=[dh.bytes_list_feature(example_data[1])]),
    standard_fields.TfExampleFields.slice_id:
    tf.train.FeatureList(feature=[dh.int64_list_feature(example_data[3])])
  }
  feature_lists = tf.train.FeatureLists(feature_list=feature_lists)

  tf_example = tf.train.SequenceExample(context=context_features,
                                        feature_lists=feature_lists)

  return tf_example.SerializeToString()


def _serialize_and_save_3d_example(elem_tuple, output_dir, split):
  elem = elem_tuple[0]
  patient_id = elem_tuple[1]
  exam_id = elem_tuple[2]
  elem_serialized = _serialize_3d_example(elem)

  writer = tf.python_io.TFRecordWriter(os.path.join(
    output_dir, split, '{}_{}.tfrecords'.format(
      patient_id, exam_id)))
  writer.write(elem_serialized)
  writer.close()


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


def _deserialize_and_decode_3d_example(
    example, target_dims, dilate_groundtruth, dilate_kernel_size,
    common_size_factor, target_depth, is_training):
  context_features = {
    standard_fields.TfExampleFields.patient_id: tf.FixedLenFeature(
      (), tf.string, default_value=''),
    standard_fields.TfExampleFields.examination_name: tf.FixedLenFeature(
      (), tf.string, default_value='')}

  feature_lists = {
    standard_fields.TfExampleFields.image_file: tf.VarLenFeature(
      tf.string),
    standard_fields.TfExampleFields.image_3d_encoded: tf.VarLenFeature(
      tf.string),
    standard_fields.TfExampleFields.annotation_3d_encoded: tf.VarLenFeature(
      tf.string),
    standard_fields.TfExampleFields.annotation_file: tf.VarLenFeature(
      tf.string),
    standard_fields.TfExampleFields.slice_id: tf.VarLenFeature(tf.int64)}

  example_dict = tf.parse_single_sequence_example(
    example, context_features=context_features,
    sequence_features=feature_lists)

  return _decode_3d_example(example_dict, target_dims=target_dims,
                            dilate_groundtruth=dilate_groundtruth,
                            dilate_kernel_size=dilate_kernel_size,
                            common_size_factor=common_size_factor,
                            target_depth=target_depth, is_training=is_training)


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


def _parse_from_exam(image_files, annotation_files,
                     patient_ids, slice_ids, exam_ids, dataset_folder):
  image_encoded = tf.map_fn(
    lambda f: tf.read_file(tf.strings.join(
      [dataset_folder, '/', f])),
    elems=image_files, parallel_iterations=util_ops.get_cpu_count())
  annotation_encoded = tf.map_fn(
    lambda f: tf.read_file(tf.strings.join(
      [dataset_folder, '/', f])),
    elems=annotation_files, parallel_iterations=util_ops.get_cpu_count())

  return (image_files, annotation_files, patient_ids, slice_ids, exam_ids,
          image_encoded, annotation_encoded)


def _build_3d_tfrecords_from_files(pickle_data, output_dir, dataset_path):
  with tf.Session() as sess:
    parse_fn = functools.partial(
            _parse_from_exam, dataset_folder=dataset_path)

    image_files_op = tf.placeholder(shape=[None], dtype=tf.string)
    annotation_files_op = tf.placeholder(shape=[None], dtype=tf.string)
    patient_ids_op = tf.placeholder(shape=[None], dtype=tf.string)
    slice_ids_op = tf.placeholder(shape=[None], dtype=tf.int64)
    exam_ids_op = tf.placeholder(shape=[None], dtype=tf.string)

    exam_data_op = (
      image_files_op, annotation_files_op, patient_ids_op, slice_ids_op,
      exam_ids_op)

    batch_size_op = tf.placeholder(shape=[], dtype=tf.int64)

    print(batch_size_op)

    dataset = tf.data.Dataset.from_tensor_slices(exam_data_op)
    dataset = dataset.batch(batch_size_op)

    dataset = dataset.map(parse_fn,
                          num_parallel_calls=util_ops.get_cpu_count())

    it = dataset.make_initializable_iterator()

    elem_op = it.get_next()

    for split, data in pickle_data[
        standard_fields.PickledDatasetInfo.data_dict].items():
      os.mkdir(os.path.join(output_dir, split))

      # Readjust data so that it is easier to create fitting tfrecords
      # Patient id to exam_id to exam_data
      # exam_data: slice id to [image, annotation, patient_id, slice_id,
      # exam_id] dict
      readjusted_data = dict()

      for patient_id, exam in data.items():
        if patient_id not in readjusted_data:
          readjusted_data[patient_id] = dict()

        for exam_id, exam_data in exam.items():
          if exam_id not in readjusted_data[patient_id]:
            readjusted_data[patient_id][exam_id] = dict()

          for elem in exam_data:
            slice_id = elem[4]
            readjusted_data[patient_id][exam_id][slice_id] = [
              elem[0], elem[1], patient_id, slice_id, elem[5]]

      serialize_and_save_fn = functools.partial(
          _serialize_and_save_3d_example, output_dir=output_dir, split=split)

      for patient_id, v in readjusted_data.items():
        for exam_id, exam_data in v.items():
          exam_entries = []

          slice_indices = list(exam_data.keys())
          slice_indices.sort()

          for slice_index in slice_indices:
            exam_entries.append(exam_data[slice_index])

          exam_data = [
            list(e) for e in list(zip(*exam_entries))]

          feed_dict = {
            image_files_op: exam_data[0], annotation_files_op: exam_data[1],
            patient_ids_op: exam_data[2], slice_ids_op: exam_data[3],
            exam_ids_op: exam_data[4], batch_size_op: len(exam_entries)}

          sess.run(it.initializer, feed_dict=feed_dict)
          elem_op_result = sess.run(elem_op, feed_dict=feed_dict)

          serialize_and_save_fn((elem_op_result, patient_id, exam_id))


def _build_regular_tfrecords_from_files(pickle_data, output_dir, dataset_path):
  with tf.Session() as sess:
    for split, data in pickle_data[
        standard_fields.PickledDatasetInfo.data_dict].items():
      os.mkdir(os.path.join(output_dir, split))

      writer_dict = dict()
      # Create writers
      for patient_id, exam in data.items():
        for exam_id, exam_data in exam.items():
          writer = tf.python_io.TFRecordWriter(os.path.join(
            output_dir, split, '{}_{}.tfrecords'.format(patient_id, exam_id)))
          writer_dict['{}_{}'.format(patient_id, exam_id)] = writer

      concatenated_elems = [
        elems for exam_id, elems in exam.items()
        for patient_id, exam in data.items()]

      dataset = tf.data.Dataset.from_tensor_slices(
        tuple([list(t) for t in zip(*list(
          itertools.chain.from_iterable(concatenated_elems)))]))

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
            writer_dict['{}_{}'.format(elem_batch_result[
              standard_fields.TfExampleFields.patient_id][i].decode(
                'utf-8'), elem_batch_result[
                  standard_fields.TfExampleFields.examination_name][i].decode(
                    'utf-8'))].write(elem_serialized)
        except tf.errors.OutOfRangeError:
          break

      for writer in writer_dict.values():
        writer.close()


def build_tfrecords_from_files(
    dataset_path, dataset_info_file,
    only_cancer_images, input_image_dims, seed, output_dir,
    tfrecords_type):
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

  if tfrecords_type == standard_fields.TFRecordsType.regular:
    _build_regular_tfrecords_from_files(
      pickle_data=pickle_data, output_dir=output_dir,
      dataset_path=dataset_path)
  elif tfrecords_type == standard_fields.TFRecordsType.input_3d:
    _build_3d_tfrecords_from_files(
      pickle_data=pickle_data, output_dir=output_dir,
      dataset_path=dataset_path)
  else:
    raise ValueError("Invalid TFRecordsType: {}".format(tfrecords_type))

  logging.info("Finished creating patient tfrecords.")


def _create_sliding_window_eval_dataset(element, target_dims, target_depth):
  # Pad image by n / 2 - 1, where n is the number of slices in input
  # Right padding needs to be + 1, since for uneven number of slices we would
  # lose the last real slice for eval
  pad_size = int(target_depth / 2 - 1)

  element[
    standard_fields.InputDataFields.image_decoded] = tf.pad(
      element[standard_fields.InputDataFields.image_decoded],
      paddings=[[pad_size, pad_size + 1], [0, 0], [0, 0], [0, 0]],
      mode='CONSTANT')
  element[
    standard_fields.InputDataFields.annotation_decoded] = tf.pad(
      element[standard_fields.InputDataFields.annotation_decoded],
      paddings=[[pad_size, pad_size + 1], [0, 0], [0, 0], [0, 0]],
      mode='CONSTANT')
  element[
    standard_fields.InputDataFields.annotation_mask] = tf.pad(
      element[standard_fields.InputDataFields.annotation_mask],
      paddings=[[pad_size, pad_size + 1], [0, 0], [0, 0], [0, 0]],
      mode='CONSTANT')

  image_decoded = tf.extract_volume_patches(
    tf.expand_dims(
      element[standard_fields.InputDataFields.image_decoded], axis=0),
    ksizes=[1, target_depth, target_dims[0], target_dims[1], 1],
    strides=[1, 2, 1, 1, 1], padding='VALID')
  image_decoded = tf.squeeze(tf.squeeze(image_decoded, axis=2), axis=0)
  image_decoded = tf.reshape(image_decoded, shape=[
    -1, target_depth, target_dims[0], target_dims[1], target_dims[2]])

  annotation_decoded = tf.extract_volume_patches(
    tf.expand_dims(
      element[standard_fields.InputDataFields.annotation_decoded], axis=0),
    ksizes=[1, target_depth, target_dims[0], target_dims[1], 1],
    strides=[1, 2, 1, 1, 1], padding='VALID')
  annotation_decoded = tf.squeeze(tf.squeeze(annotation_decoded, axis=2),
                                  axis=0)
  annotation_decoded = tf.reshape(annotation_decoded, shape=[
    -1, target_depth, target_dims[0], target_dims[1], 3])

  annotation_mask = tf.extract_volume_patches(
    tf.expand_dims(
      element[standard_fields.InputDataFields.annotation_mask], axis=0),
    ksizes=[1, target_depth, target_dims[0], target_dims[1], 1],
    strides=[1, 2, 1, 1, 1], padding='VALID')
  annotation_mask = tf.squeeze(tf.squeeze(annotation_mask, axis=2), axis=0)
  annotation_mask = tf.reshape(annotation_mask, shape=[
    -1, target_depth, target_dims[0], target_dims[1], 1])

  def sliding_window_1d(e, is_string):
    if is_string:
      indices = tf.range(0, tf.shape(e)[0])
      indices = tf.pad(indices, [[pad_size, pad_size + 1]],
                       constant_values=-1)
      indices = tf.expand_dims(
        tf.expand_dims(tf.expand_dims(indices, axis=0), 0), -1)
      indices = tf.image.extract_image_patches(
        indices, ksizes=[1, 1, target_depth, 1], strides=[1, 1, 2, 1],
        rates=[1, 1, 1, 1], padding='VALID')
      indices = tf.squeeze(tf.squeeze(indices, axis=0), axis=0)

      indices = tf.expand_dims(indices, axis=-1)

      valid_indices = tf.where(tf.equal(indices, -1),
                               tf.zeros_like(indices), indices)
      e = tf.gather_nd(e, valid_indices)

      indices = tf.squeeze(indices, axis=-1)

      e = tf.where(
        tf.equal(indices, -1), tf.tile(
          tf.expand_dims(tf.constant(['']), axis=0),
          multiples=tf.shape(e)), e)

      return e

    else:
      e = tf.pad(e, [[pad_size, pad_size + 1]],
                 constant_values='' if is_string else -1)
      e = tf.expand_dims(
        tf.expand_dims(tf.expand_dims(e, axis=0), 0), -1)
      e = tf.image.extract_image_patches(
        e, ksizes=[1, 1, target_depth, 1], strides=[1, 1, 2, 1],
        rates=[1, 1, 1, 1], padding='VALID')
      e = tf.squeeze(tf.squeeze(e, axis=0), axis=0)

      return e

  slice_ids = sliding_window_1d(
    element[standard_fields.InputDataFields.slice_id], is_string=False)
  image_files = sliding_window_1d(
    element[standard_fields.InputDataFields.image_file], is_string=True)
  annotation_files = sliding_window_1d(
    element[standard_fields.InputDataFields.annotation_file], is_string=True)

  patient_id = tf.tile(
    tf.expand_dims(element[standard_fields.InputDataFields.patient_id],
                   axis=0), multiples=[tf.shape(slice_ids)[0]])
  exam_name = tf.tile(tf.expand_dims(element[
    standard_fields.InputDataFields.examination_name], axis=0),
                      multiples=[tf.shape(slice_ids)[0]])

  result_dict = {
    standard_fields.InputDataFields.patient_id: patient_id,
    standard_fields.InputDataFields.image_decoded: image_decoded,
    standard_fields.InputDataFields.annotation_decoded:
    annotation_decoded,
    standard_fields.InputDataFields.annotation_mask:
    annotation_mask,
    standard_fields.InputDataFields.examination_name: exam_name,
    standard_fields.InputDataFields.image_file: image_files,
    standard_fields.InputDataFields.annotation_file: annotation_files,
    standard_fields.InputDataFields.slice_id: slice_ids}

  dataset = tf.data.Dataset.from_tensor_slices(result_dict)

  return dataset


def build_tf_dataset_from_tfrecords(directory, split_name, target_dims,
                                    patient_ids, is_training,
                                    dilate_groundtruth, dilate_kernel_size,
                                    common_size_factor, model_objective,
                                    tfrecords_type, target_depth):
  tfrecords_folder = os.path.join(directory, 'tfrecords', split_name)
  assert(os.path.exists(tfrecords_folder))

  tfrecords_files = os.listdir(tfrecords_folder)
  tfrecords_files = [os.path.join(tfrecords_folder, file_name)
                     for file_name in tfrecords_files]

  if model_objective == 'segmentation':
    if tfrecords_type == 'regular':
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
          lambda tfrecords_file: tf.data.TFRecordDataset(
            tfrecords_file).shuffle(
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
    elif tfrecords_type == 'input_3d':
      dataset = tf.data.Dataset.from_tensor_slices(tfrecords_files)

      if is_training:
        dataset = dataset.interleave(
          lambda tfrecords_file: tf.data.TFRecordDataset(
            tfrecords_file).repeat(None), block_length=1,
          cycle_length=len(tfrecords_files),
          num_parallel_calls=util_ops.get_cpu_count())
      else:
        dataset = dataset.interleave(
          lambda tfrecords_file: tf.data.TFRecordDataset(
            tfrecords_file), block_length=1, cycle_length=len(tfrecords_files),
          num_parallel_calls=util_ops.get_cpu_count())

      deserialize_and_decode_fn = functools.partial(
        _deserialize_and_decode_3d_example, target_dims=target_dims,
        dilate_groundtruth=dilate_groundtruth,
        dilate_kernel_size=dilate_kernel_size,
        common_size_factor=common_size_factor,
        target_depth=target_depth, is_training=is_training)

      dataset = dataset.map(deserialize_and_decode_fn,
                            num_parallel_calls=util_ops.get_cpu_count())

      if not is_training:
        dataset = dataset.flat_map(
          lambda e: _create_sliding_window_eval_dataset(
            e, target_dims=target_dims, target_depth=target_depth))

      return dataset
    else:
      raise ValueError("Invalid tfrecords type.")

  elif model_objective == 'interpolation':
    tfrecords_dict = dict()
    for tfrecords_file in tfrecords_files:
      patient_id, exam_name = os.path.splitext(
        os.path.basename(tfrecords_file))[0].split('_')
      if patient_id not in tfrecords_dict:
        tfrecords_dict[patient_id] = []
      tfrecords_dict[patient_id].append(tfrecords_file)

    tfrecords_files = list(tfrecords_dict.values())

    dataset = tf.data.Dataset.from_tensor_slices(list(zip(*tfrecords_files)))

    deserialize_and_decode_fn = functools.partial(
        _deserialize_and_decode_example, target_dims=target_dims,
        dilate_groundtruth=dilate_groundtruth,
        dilate_kernel_size=dilate_kernel_size,
        common_size_factor=common_size_factor)

    def make_exam_tfrecords(patient_exam_tfrecords):
      tfrecords_datasets = []
      patient_exam_tfrecords = tf.unstack(patient_exam_tfrecords, axis=0)
      for tfrecords_file in patient_exam_tfrecords:
        tfrecords_dataset = tf.data.TFRecordDataset(tfrecords_file).map(
          deserialize_and_decode_fn,
          num_parallel_calls=util_ops.get_cpu_count())
        # Make sure the elements are grouped for interpolation, i.e. we want to
        # have 3 slices in a row at all times
        tfrecords_dataset = tfrecords_dataset.window(size=3, shift=1).flat_map(
          lambda x: x.batch(3))

        if is_training:
          # Add repeat here since otherwise patients with many slices would
          # have a bias
          tfrecords_dataset = tfrecords_dataset.repeat(None)

        tfrecords_datasets.append(tfrecords_dataset)

      # Merge all
      final_dataset = tfrecords_datasets[0]
      for i in range(1, len(tfrecords_datasets)):
        final_dataset = final_dataset.concatenate(tfrecords_datasets[i])

      return final_dataset

    dataset = dataset.interleave(make_exam_tfrecords,
                                 block_length=1,
                                 cycle_length=len(tfrecords_files))

    return dataset
  else:
    raise ValueError("Unknown model objective: {}".format(model_objective))


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
