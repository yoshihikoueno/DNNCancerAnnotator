'''
this module will provide abstraction layer for the dataset API


Expected directory structure
- data_dir -- train -- cancer -- patientID -- examID -- TRA -- 01.jpg 02.jpg ...
           |        |                                |- ADC -- 01.jpg 02.jpg ...
           |        |                                |- DWI -- 01.jpg 02.jpg ...
           |        |                                |- DCEE -- 01.jpg 02.jpg ...
           |        |                                |- DCEL -- 01.jpg 02.jpg ...
           |        |                                |- label -- 01.jpg 02.jpg ...
           |        |
           |        |- healthy -- patientID -- examID -- TRA -- 01.jpg 02.jpg ...
           |                                          |- ADC -- 01.jpg 02.jpg ...
           |                                          |- DWI -- 01.jpg 02.jpg ...
           |                                          |- DCEE -- 01.jpg 02.jpg ...
           |
           |- val -- cancer -- patientID -- examID -- TRA -- 01.jpg 02.jpg ...
           |      |                                |- ADC -- 01.jpg 02.jpg ...
           |      |                                |- DWI -- 01.jpg 02.jpg ...
           |      |                                |- DCEE -- 01.jpg 02.jpg ...
           |      |                                |- DCEL -- 01.jpg 02.jpg ...
           |      |                                |- label -- 01.jpg 02.jpg ...
           |      |
           |      |- healthy -- patientID -- examID -- TRA -- 01.jpg 02.jpg ...
           |                                        |- ADC -- 01.jpg 02.jpg ...
           |                                        |- DWI -- 01.jpg 02.jpg ...
           |                                        |- DCEE -- 01.jpg 02.jpg ...
           |
           |- test -- cancer -- patientID -- examID -- TRA -- 01.jpg 02.jpg ...
                   |                                |- ADC -- 01.jpg 02.jpg ...
                   |                                |- DWI -- 01.jpg 02.jpg ...
                   |                                |- DCEE -- 01.jpg 02.jpg ...
                   |                                |- DCEL -- 01.jpg 02.jpg ...
                   |                                |- label -- 01.jpg 02.jpg ...
                   |
                   |- healthy -- patientID -- examID -- TRA -- 01.jpg 02.jpg ...
                                                     |- ADC -- 01.jpg 02.jpg ...
                                                     |- DWI -- 01.jpg 02.jpg ...
                                                     |- DCEE -- 01.jpg 02.jpg ...
'''

# built in
import os
import pdb
from glob import glob
from functools import partial

# external
import tensorflow as tf
from tqdm import tqdm

# customs
from .utils import dataset as ds_utils


def train_ds(
    path,
    batch_size,
    buffer_size,
    repeat=True,
    slice_types=('TRA', 'ADC', 'DWI', 'DCEE', 'DCEL', 'label'),
    normalize_exams=True,
):
    '''
    generate dataset for training

    Args:
        path: train data path
        batch_size: batch size
        buffer_size: buffer_size
        repeat: should ds be repeated
        slice_types: types of slices to include
        normalize_exams: whether the resulting dataset contain
            the same number of slices from each exam
    '''
    ds = base(path, slice_types=slice_types, normalize_exams=normalize_exams)
    ds = augment(ds)
    ds = to_feature_label(ds, slice_types=slice_types)
    ds = ds.shuffle(buffer_size)
    if repeat:
        ds = ds.repeat(None)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def eval_ds(path, batch_size):
    '''
    generate dataset for evaluation
    '''
    ds = base(path)
    ds = to_feature_label(ds)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def predict_ds(path):
    '''
    generate dataset for prediction
    '''
    ds = base(path)
    ds = to_feature_label(ds)
    ds = ds.batch(1)
    return ds


def base(path, slice_types, output_size=(512, 512), dtype=tf.float32, normalize_exams=True):
    '''
    generate base dataset
    '''
    if os.path.splitext(path)[1] == '.tfrecords':
        ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
        ds = ds.map(
            lambda x: tf.io.parse_single_example(x, {
                'slices': tf.io.FixedLenFeature([], tf.string),
                'patientID': tf.io.FixedLenFeature([], tf.int64),
                'examID': tf.io.FixedLenFeature([], tf.int64),
                'path': tf.io.FixedLenFeature([], tf.string),
                'category': tf.io.FixedLenFeature([], tf.string),
                'shape': tf.io.FixedLenFeature([4], tf.int64),
            })
        )
        ds = ds.map(lambda x: {
            'slices': tf.reshape(tf.io.parse_tensor(x['slices'], tf.uint8), x['shape']),
            'patientID': x['patientID'],
            'examID': x['examID'],
            'path': x['path'],
            'category': x['category'],
        })
        if normalize_exams:
            cancer_ds = ds.filter(lambda x: x['category'] == 'cancer')
            cancer_ds = cancer_ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x['slices'])).repeat(None)
            healthy_ds = ds.filter(lambda x: x['category'] == 'healthy')
            healthy_ds = healthy_ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x['slices'])).repeat(None)
            ds = tf.data.Dataset.from_tensor_slices([0, 1]).interleave(
                lambda x: cancer_ds if x == 1 else healthy_ds,
                cycle_length=2,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        else: ds = ds.map(lambda x: tf.data.Dataset.from_tensor_slices(x['slices']))
    else:
        assert os.path.isdir(path)
        pattern = os.path.join(path, *'*' * 3)
        ds = tf.data.Dataset.list_files(pattern)
        ds = ds.interleave(
            partial(
                tf_prepare_combined_slices,
                slice_types=slice_types,
                return_type='infinite_dataset' if normalize_exams else 'dataset',
            ),
            cycle_length=ds_utils.count(ds),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    ds = ds.map(
        lambda image: tf.image.crop_to_bounding_box(
            image,
            ((tf.shape(image)[:2] - output_size) // 2)[0],
            ((tf.shape(image)[:2] - output_size) // 2)[1],
            *output_size,
        ),
        tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda x: tf.reshape(x, [*x.shape[:-1], len(slice_types)]), tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda x: tf.cast(x, dtype=dtype), tf.data.experimental.AUTOTUNE)
    return ds


def generate_tfrecords(path, output, slice_types=('TRA', 'ADC', 'DWI', 'DCEE', 'DCEL', 'label')):
    '''
    Generate TFRecords

    Args:
        path: path to the data directory
        output: output path
        slice_types: list of slices to be included
    '''
    pattern = os.path.join(path, *'*' * 3)
    exams = glob(pattern)
    with tf.io.TFRecordWriter(output, 'GZIP') as writer:
        for exam in tqdm(exams, 'Generating TFRecords'):
            exam_data = prepare_combined_slices(exam, slice_types)
            example = tf.train.Example(features=tf.train.Features(feature={
                'slices': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(exam_data['slices']).numpy()])),
                'patientID': tf.train.Feature(int64_list=tf.train.Int64List(value=[exam_data['patientID']])),
                'examID': tf.train.Feature(int64_list=tf.train.Int64List(value=[exam_data['examID']])),
                'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[exam_data['path'].encode()])),
                'category': tf.train.Feature(bytes_list=tf.train.BytesList(value=[exam_data['category'].encode()])),
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=exam_data['slices'].shape)),
            }))
            writer.write(example.SerializeToString())
    return


def tf_prepare_combined_slices(exam_dir, slice_types, return_type='array'):
    return_type = return_type.lower()
    if return_type == 'array':
        return tf.py_function(
            lambda x: partial(prepare_combined_slices, slice_types=slice_types)(x)['slices'],
            [exam_dir],
            tf.uint8,
        )
    elif return_type == 'dataset':
        return tf.data.Dataset.from_tensor_slices(
            tf_prepare_combined_slices(exam_dir, slice_types=slice_types, return_type='array')
        )
    elif return_type == 'infinite_dataset':
        return tf_prepare_combined_slices(exam_dir, slice_types=slice_types, return_type='dataset').repeat(None)
    else: raise NotImplementedError


def prepare_combined_slices(exam_dir, slice_types):
    if isinstance(exam_dir, str): pass
    elif isinstance(exam_dir, tf.Tensor): exam_dir = exam_dir.numpy().decode()
    else: raise NotImplementedError
    exam_data = parse_exam(exam_dir, slice_types=slice_types)
    slice_names = exam_data['TRA'].keys()

    slices = tf.stack([tf.stack([exam_data[type_][slice_] for type_ in slice_types], axis=-1) for slice_ in slice_names])
    return dict(
        slices=slices,
        category=exam_data['category'],
        patientID=exam_data['patientID'],
        examID=exam_data['examID'],
        path=exam_data['path'],
    )


def get_category_from_exam_path(exam_dir):
    category = exam_dir.split(os.path.sep)[-3]
    assert category in ('healthy', 'cancer'), f'Unknown category {category}: {exam_dir}'
    return category


def parse_exam(exam_dir, slice_types, decoder=tf.image.decode_image):
    '''
    parse exam directory and return contents in dict

    Args:
        exam_dir: exam path
        slice_types: list of slice types to consider
        decoder: image decoder

    Returns:
        dict: {
            'category': 'cancer' or 'healthy',
            'path': exam_dir,
            'patientID': patient ID,
            'examID': exam ID,
            'nslices': the number of available slices,
            'label': segmentation map
            'TRA': decoded TRA slices,
            'ADC': decoded ADC slices,
            'DWI': decoded DWI slices,
            'DCEE': decoded DCEE slices,
            'DCEL': decoded DCEL slices,
        }
    '''
    result = {'path': exam_dir}
    result['category'] = get_category_from_exam_path(exam_dir)
    result['patientID'], result['examID'] = getID_from_exam_path(exam_dir)

    if result['category'] == 'cancer':
        slices_per_type = {
            slice_type: set(os.listdir(os.path.join(exam_dir, slice_type)))
            for slice_type in slice_types
        }
    elif result['category'] == 'healthy':
        slices_per_type = {
            slice_type: set(os.listdir(os.path.join(exam_dir, slice_type)))
            for slice_type in slice_types if slice_type != 'label'
        }
        slices_per_type['label'] = slices_per_type['TRA']
    else:
        raise NotImplementedError

    common_slices = set.intersection(*map(
        lambda slices: set(
            map(lambda name: os.path.splitext(name)[0], slices)),
        slices_per_type.values(),
    ))
    assert common_slices, f'Not enough slices in {exam_dir}'

    for slice_type in slice_types:
        slices_per_type[slice_type] = list(
            filter(lambda x: os.path.splitext(x)[0], slices_per_type[slice_type]))

    result['nslices'] = len(common_slices)

    def wrapper(func):
        def _func(x):
            # print(x)
            x = tf.io.read_file(x)
            return func(x)
        return _func
    decoder = wrapper(decoder)

    for slice_type, names in slices_per_type.items():
        if (slice_type == 'label') and (result['category'] == 'healthy'):
            result[slice_type] = {
                os.path.splitext(name)[0]: tf.zeros_like(decoder(os.path.join(exam_dir, 'TRA', name)))[:, :, 0]
                for name in slices_per_type['TRA']
            }
        else:
            result[slice_type] = {
                os.path.splitext(name)[0]: decoder(os.path.join(exam_dir, slice_type, name))[:, :, 0] for name in names
            }
    return result


def getID_from_exam_path(exam_path):
    '''
    parse patientID and examID from given exam_path
    '''
    patient_id, exam_id = map(int, os.path.normpath(
        exam_path).strip(os.path.sep).split(os.path.sep)[-2:])
    return patient_id, exam_id


def base_from_tfrecords(path):
    raise NotImplementedError
    return


def augment(ds, methods=None):
    return ds


def to_feature_label(ds, slice_types):
    '''
    convert ds containing dicts to ds containing tuple(feature, tuple)
    '''
    feature_slice_indices = [i for i in range(len(slice_types)) if slice_types[i] != 'label']
    label_index = slice_types.index('label')

    def convert(combined_slices):
        feature = tf.gather(combined_slices, feature_slice_indices, axis=-1)
        label = combined_slices[:, :, label_index]
        return feature, label

    ds = ds.map(convert, tf.data.experimental.AUTOTUNE)
    return ds
