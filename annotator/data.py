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
import p_tqdm
import tensorflow_addons as tfa

# customs
from .utils import dataset as ds_utils


_TFRECORD_COMPRESSION = None


def train_ds(
    path,
    batch_size,
    buffer_size,
    repeat=True,
    slice_types=('TRA', 'ADC', 'DWI', 'DCEE', 'DCEL', 'label'),
    normalize_exams=True,
    output_size=(256, 256),
    augment_options=None,
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
        output_size: size of images in the dataset
            images will be centrally cropped to match the size
        augment_options (dict): options for augmentation
    '''
    if augment_options is None: augment_options = {'random_crop': {}}
    default_augment_options = {
        augment_random_crop: dict(output_size=output_size),
        augment_random_flip: {},
        augment_random_contrast: dict(target_channels=list(range(len(slice_types[:-1])))),
        augment_random_warp: {},
    }

    ds = base(
        path,
        output_size=(512, 512),
        slice_types=slice_types,
        normalize_exams=normalize_exams,
    )
    ds = augment(
        ds,
        methods=parse_augment_options(augment_options, default_augment_options),
    )
    ds = to_feature_label(ds, slice_types=slice_types)
    ds = ds.shuffle(buffer_size)
    if repeat:
        ds = ds.repeat(None)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def eval_ds(
    path,
    batch_size,
    slice_types=('TRA', 'ADC', 'DWI', 'DCEE', 'DCEL', 'label'),
    include_meta=False,
    output_size=(512, 512),
):
    '''
    generate dataset for training

    Args:
        path: train data path
        batch_size: batch size
        include_meta: whether output ds should contain meta info
        slice_types: types of slices to include
        normalize_exams: whether the resulting dataset contain
            the same number of slices from each exam
        output_size: size of images in the dataset
            images will be centrally cropped to match the size
    '''
    ds = base(
        path,
        slice_types=slice_types,
        normalize_exams=False,
        include_meta=include_meta,
        output_size=output_size,
    )
    ds = to_feature_label(ds, slice_types=slice_types, include_meta=include_meta)
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


def base(path, slice_types, output_size=(512, 512), dtype=tf.float32, normalize_exams=True, include_meta=False):
    '''
    generate base dataset
    '''
    if not isinstance(path, list): path = list(path)
    if os.path.splitext(path[0])[1] == '.tfrecords':
        assert all(map(lambda x: os.path.splitext(x)[1] == '.tfrecords', path))

        ds = base_from_tfrecords(path, normalize=normalize_exams, include_meta=include_meta, output_slice_types=slice_types)
    else:
        assert all(map(os.path.isdir, path))
        pattern = list(map(lambda x: os.path.join(x, *'*' * 3), path))
        ds = tf.data.Dataset.from_tensor_slices(pattern)
        ds = ds.interleave(tf.data.Dataset.list_files)
        ds = ds.interleave(
            partial(
                tf_prepare_combined_slices,
                slice_types=slice_types,
                return_type='infinite_dataset' if normalize_exams else 'dataset',
            ),
            cycle_length=ds_utils.count(ds),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    if include_meta:
        tf.data.Dataset.partial_map = partial_map
        if output_size is not None: ds = ds.partial_map(
            'slice',
            lambda image: tf.image.crop_to_bounding_box(
                image,
                ((tf.shape(image)[:2] - output_size) // 2)[0],
                ((tf.shape(image)[:2] - output_size) // 2)[1],
                *output_size,
            ),
        )
        ds = ds.partial_map('slice', lambda x: tf.reshape(x, [*x.shape[:-1], len(slice_types)]))
        ds = ds.partial_map('slice', lambda x: tf.cast(x, dtype=dtype))
        ds = ds.partial_map('slice', lambda x: x / 255.0)
    else:
        if output_size is not None: ds = ds.map(
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
        ds = ds.map(lambda x: x / 255.0, tf.data.experimental.AUTOTUNE)
    return ds


def partial_map(ds, key, func):
    def wrapped_func(data):
        data.update({key: func(data[key])})
        return data
    ds = ds.map(wrapped_func, tf.data.experimental.AUTOTUNE)
    return ds


def generate_tfrecords(
    path,
    output,
    category=None,
    slice_types=('TRA', 'ADC', 'DWI', 'DCEE', 'DCEL', 'label'),
    output_size=(512, 512),
):
    '''
    Generate TFRecords

    Args:
        path: path to the data directory
            under the path, it should be structured as below:
            -- path -- healthy -- patientID -- examID -- slice_type -- <sliceID>.png
                    |- cancer -- patientID -- examID -- slice_type -- <sliceID>.png
        output: output path
        category: category to include (e.g. cancer or healthy)
            default (None): include all
        slice_types: list of slices to be included
    '''
    def serialize(slices, patientID, examID, path, category):
        serialized = tf.py_function(
            lambda slices, patientID, examID, path, category:
                tf.train.Example(features=tf.train.Features(feature={
                    'slices': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(slices).numpy()])),
                    'patientID': tf.train.Feature(int64_list=tf.train.Int64List(value=[patientID])),
                    'examID': tf.train.Feature(int64_list=tf.train.Int64List(value=[examID])),
                    'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[path.numpy()])),
                    'category': tf.train.Feature(bytes_list=tf.train.BytesList(value=[category.numpy()])),
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=slices.shape)),
                    'slice_types': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=list(map(lambda x: x.encode(), slice_types)))),
                })).SerializeToString(),
            (slices, patientID, examID, path, category),
            tf.string,
        )
        return serialized

    pattern = os.path.join(path, *'*' * 3)
    exams = glob(pattern)
    ds = tf.data.Dataset.from_generator(
        lambda: tqdm(map(partial(prepare_combined_slices, slice_types=slice_types), exams), total=len(exams)),
        output_types={
            'slices': tf.uint8,
            'patientID': tf.int64,
            'examID': tf.int64,
            'category': tf.string,
            'path': tf.string,
        },
    )
    ds = ds.map(
        lambda exam_data: {
            'slices': tf.map_fn(
                lambda image: tf.image.crop_to_bounding_box(
                    image,
                    ((tf.shape(image)[:2] - output_size) // 2)[0],
                    ((tf.shape(image)[:2] - output_size) // 2)[1],
                    *output_size,),
                exam_data['slices'],
            ),
            'patientID': exam_data['patientID'],
            'examID': exam_data['examID'],
            'category': exam_data['category'],
            'path': exam_data['path'],
        },
        tf.data.experimental.AUTOTUNE,
    )
    if category is not None: ds = ds.filter(lambda x: x['category'] == category)
    ds = ds.map(
        lambda exam_data: serialize(
            exam_data['slices'],
            exam_data['patientID'],
            exam_data['examID'],
            exam_data['path'],
            exam_data['category'],
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    writer = tf.data.experimental.TFRecordWriter(output, _TFRECORD_COMPRESSION)
    writer.write(ds)
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


def prepare_combined_slices(exam_dir, slice_types, shape_variance_torelance=0.007):
    if isinstance(exam_dir, str): pass
    elif isinstance(exam_dir, tf.Tensor): exam_dir = exam_dir.numpy().decode()
    else: raise NotImplementedError
    exam_data = parse_exam(exam_dir, slice_types=slice_types)
    slice_names = exam_data['TRA'].keys()

    shapes = tf.stack(list(exam_data[t][s].shape for t in slice_types for s in slice_names), 0)
    shape_min, shape_max = tf.reduce_min(shapes, 0), tf.reduce_max(shapes, 0)
    shape_diff = (shape_max - shape_min) / tf.reduce_mean(shapes, 0)
    if (tf.reduce_max(shape_diff) > shape_variance_torelance):
        raise ValueError(
            'Shape of input image differs greately.\n'
            f'Exam: {exam_dir}\n'
            f'Shapes: {shapes}'
        )
    slices = tf.stack([
        tf.stack([exam_data[type_][slice_][:shape_min[0], :shape_min[1]] for type_ in slice_types], axis=-1)
        for slice_ in slice_names
    ])
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


def extract_slices_from_tfrecord(path, output_slice_types=None, include_meta=True):
    '''
    extract data from tfrecords

    Args:
        path: path to the tfrecord file
        output_slice_types: (optional) list of slices included in output
            default (None): output what is recorded in tfrecord file
        include_meta: whether output should include meta information in addition to slices
    '''
    ds = tf.data.TFRecordDataset(path, compression_type=_TFRECORD_COMPRESSION)
    ds = ds.map(
        lambda x: tf.io.parse_single_example(x, {
            'slices': tf.io.FixedLenFeature([], tf.string),
            'patientID': tf.io.FixedLenFeature([], tf.int64),
            'examID': tf.io.FixedLenFeature([], tf.int64),
            'path': tf.io.FixedLenFeature([], tf.string),
            'category': tf.io.FixedLenFeature([], tf.string),
            'shape': tf.io.FixedLenFeature([4], tf.int64),
            'slice_types': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        }),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(
        lambda x: {
            'slices': tf.reshape(tf.io.parse_tensor(x['slices'], tf.uint8), x['shape']),
            'patientID': x['patientID'],
            'examID': x['examID'],
            'path': x['path'],
            'category': x['category'],
            'slice_types': tf.reshape(x['slice_types'], [x['shape'][-1]]),
        },
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    def get_indices(slice_types, output_slice_types):
        indices = tf.squeeze(tf.map_fn(lambda x: tf.where(x == slice_types), tf.constant(output_slice_types), dtype=tf.int64))
        return indices

    ds = ds.map(
        lambda x: {
            'slices': tf.gather(x['slices'], get_indices(x['slice_types'], output_slice_types), axis=3),
            'patientID': x['patientID'],
            'examID': x['examID'],
            'path': x['path'],
            'category': x['category'],
            'slice_types': tf.gather(x['slice_types'], get_indices(x['slice_types'], output_slice_types)),
        },
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if include_meta:
        ds = ds.flat_map(
            lambda x: tf.data.Dataset.zip(
                (
                    tf.data.Dataset.from_tensor_slices(x['slices']),
                    tf.data.Dataset.from_tensors(x['patientID']).repeat(None),
                    tf.data.Dataset.from_tensors(x['examID']).repeat(None),
                    tf.data.Dataset.from_tensors(x['path']).repeat(None),
                    tf.data.Dataset.from_tensors(x['category']).repeat(None),
                    tf.data.Dataset.from_tensors(x['slice_types']).repeat(None),
                    tf.data.experimental.Counter(),
                )
            ))
        ds = ds.map(lambda slice_, patientID, examID, path, category, slice_types, sliceID: {
            'slice': slice_,
            'patientID': patientID,
            'examID': examID,
            'path': path,
            'category': category,
            'slice_types': slice_types,
            'sliceID': sliceID,
        }, tf.data.experimental.AUTOTUNE)
    else:
        ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x['slices']))
    return ds


def base_from_tfrecords(path: list, normalize=False, include_meta=False, output_slice_types=None):
    ds = tf.data.Dataset.from_tensor_slices(path)
    if normalize:
        ds = ds.interleave(
            lambda path: extract_slices_from_tfrecord(
                path,
                output_slice_types=output_slice_types,
                include_meta=include_meta,
            ).repeat(None),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    else:
        ds = ds.interleave(
            lambda path: extract_slices_from_tfrecord(
                path,
                output_slice_types=output_slice_types,
                include_meta=include_meta,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    return ds


def parse_augment_options(options, default_options=None):
    '''parse augment options'''
    if default_options is None: default_options = {}
    result = {}
    for name, conf in options.items():
        if conf is None: conf = {}
        func = globals()[f'augment_{name}']
        if func in default_options:
            new_conf = default_options[func].copy()
            new_conf.update(conf)
            conf = new_conf

        result[func] = conf
    return result


def augment(ds, methods=None):
    if methods is None:
        methods = {
            augment_random_crop: {},
            augment_random_flip: {},
            augment_random_contrast: {},
            augment_random_warp: {},
        }
    else:
        assert isinstance(methods, dict)
        methods = dict(map(
            lambda name, config: (solve_augment_method(name), config),
            methods.keys(), methods.values(),
        ))

    for op, config in methods.items():
        print('Augment: applying', op, config)
        ds = op(ds, **config)
    return ds


def solve_augment_method(method_str):
    '''
    check if the specified augment method exists
    and if it's really an augment method.
    '''
    if callable(method_str): return method_str
    method_str.startswith('augment_')
    method = vars[method_str]
    return method


def augment_random_contrast(ds, target_channels, lower=0.8, upper=1.2):
    ds = ds.map(
        lambda image: random_contrast(image, lower=lower, upper=upper, target_channels=target_channels),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def random_contrast(image, lower, upper, target_channels):
    non_target_channels = [i for i in range(image.shape[-1]) if i not in target_channels]

    target = tf.gather(image, target_channels, axis=2)
    non_target = tf.gather(image, non_target_channels, axis=2)
    target_out = tf.image.random_contrast(target, lower=lower, upper=upper)
    image = tf.concat([target_out, non_target], axis=2)
    indices = list(map(
        lambda xy: xy[1],
        sorted(
            zip(target_channels + non_target_channels, range(1000)),
            key=lambda xy: xy[0],
        ),
    ))
    image = tf.gather(image, indices, axis=2)
    return image


def augment_random_hue(ds, max_delta=0.2):
    ds = ds.map(
        lambda image: tf.image.random_hue(image, max_delta=max_delta),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def augment_random_flip(ds):
    ds = ds.map(
        tf.image.random_flip_left_right,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def augment_random_warp(ds: tf.data.Dataset, process_in_batch=10, **options) -> tf.data.Dataset:
    '''apply augmentation based on image warping

    Args:
        process_in_batch: the number of images to apply warping in a batch
            None to disable this feature
        options: options to be passed to random_warp function
    '''
    if process_in_batch is not None:
        ds = ds.batch(process_in_batch)
    ds = ds.map(
        lambda image: random_warp(image, process_in_batch=process_in_batch, **options),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if process_in_batch is not None:
        ds = ds.unbatch()
    return ds


def augment_random_crop(ds, **options):
    '''apply augmentation based on image warping'''
    ds = ds.map(
        lambda image: random_crop(image, **options),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def augment_random_intrachannelwarp(ds: tf.data.Dataset, paired=((0, -1),), **options) -> tf.data.Dataset:
    '''warps each channel indenpently.

    As oppposed to `augment_random_warp`, this will introduce misalignments between
    channels. For this reason, this function is not intended to used as a regular training,
    rather for experiment purpose to see how a model robust against misalignments between slices.

    Args:
        paired (list[tuple[int, int]]): list of paired channels. Paired channels will be applied
            warping operation together, keeping them consistent.
            This is meant to be used to align a channel to a label, while still introducing
            artificial misalignments between channels.
        options: options to be passed to random_warp function
    '''
    ds = ds.map(
        lambda image: random_intrachannelwarp(image, paired=paired, **options),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def random_crop(image, output_size=(512, 512), stddev=4, max_=6, min_=-6):
    """
    performs augmentation by cropping/resizing
    given image
    """
    diff = tf.clip_by_value(tf.cast(tf.random.normal([2], stddev=stddev), tf.int32), min_, max_)
    image = tf.image.crop_to_bounding_box(
        image,
        ((tf.shape(image)[:2] - output_size) // 2 + diff)[0],
        ((tf.shape(image)[:2] - output_size) // 2 + diff)[1],
        *output_size,
    )
    return image


def random_intrachannelwarp(image, n_points=100, max_diff=5, stddev=2.0, paired=()):
    paired = list(map(
        lambda channel_list: list(map(
            lambda channel: channel if channel >= 0 else image.get_shape()[-1] + channel,
            channel_list,
        )),
        paired,
    ))
    non_paired = list(map(
        lambda x: [x],
        set(range(image.get_shape()[-1])) - set([index for indices in paired for index in indices]),
    ))
    channel_groups = paired + non_paired
    warped_groups = list(map(
        lambda group: random_warp(tf.gather(image, group, axis=-1), n_points=100, max_diff=max_diff, stddev=stddev),
        channel_groups,
    ))
    warped_groups = [channel for channels in warped_groups for channel in tf.unstack(channels, axis=-1)]
    indices = [index for indices in channel_groups for index in indices]
    image = tf.stack(
        list(map(lambda x: x[1], sorted(zip(indices, warped_groups), key=lambda x: x[0]))),
        axis=-1
    )
    return image


@tf.function
def random_warp(image, n_points=100, max_diff=5, stddev=2.0, process_in_batch=None):
    '''
    this function will perfom data augmentation
    using Non-affine transformation, namely
    image warping.
    Currently, only square images are supported

    Args:
        image: input image
        n_points: the num of points to take for image warping
        max_diff: maximum movement of pixels
    Return:
        warped image
    '''
    if process_in_batch is not None:
        width_index, height_index, n_images = 1, 2, process_in_batch
        image = tf.reshape(image, [n_images, *image.get_shape()[1:]])
    else:
        width_index, height_index, n_images = 0, 1, 1

    width = tf.shape(image)[width_index]
    height = tf.shape(image)[height_index]

    with tf.control_dependencies([tf.assert_equal(width, height)]):
        raw = tf.random.uniform([n_images, n_points, 2], 0.0, tf.cast(width, tf.float32), dtype=tf.float32)
        diff = tf.random.normal([n_images, n_points, 2], mean=0.0, stddev=stddev, dtype=tf.float32)
        # ensure that diff is not too big
        diff = tf.clip_by_value(diff, tf.cast(-max_diff, tf.float32), tf.cast(max_diff, tf.float32))

    if process_in_batch is None:
        # expand dimension to meet the requirement of sparse_image_warp
        image = tf.expand_dims(image, 0)

    image = tfa.image.sparse_image_warp(
        image=image,
        source_control_point_locations=raw,
        dest_control_point_locations=raw + diff,
    )[0]
    # sparse_image_warp function will return a tuple
    # (warped image, flow_field)

    if process_in_batch is None:
        # shrink dimension
        image = image[0, :, :, :]
    return image


def to_feature_label(ds, slice_types, include_meta=False):
    '''
    convert ds containing dicts to ds containing tuple(feature, tuple)
    '''
    feature_slice_indices = [i for i in range(len(slice_types)) if slice_types[i] != 'label']
    label_index = slice_types.index('label')

    def convert(data):
        if include_meta:
            combined_slices = data['slice']
            feature = tf.gather(combined_slices, feature_slice_indices, axis=-1)
            label = combined_slices[:, :, label_index]
            data.update({'x': feature, 'y': label})
            data.pop('slice')
            return data
        else:
            combined_slices = data
            feature = tf.gather(combined_slices, feature_slice_indices, axis=-1)
            label = combined_slices[:, :, label_index]
            return feature, label

    ds = ds.map(convert, tf.data.experimental.AUTOTUNE)
    return ds
