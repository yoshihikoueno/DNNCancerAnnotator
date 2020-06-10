'''
this module will provide abstraction layer for the dataset API


Expected directory structure
- data_dir -- train -- patientID -- examID -- TRA -- 01.jpg 02.jpg ...
           |                               |- ADC -- 01.jpg 02.jpg ...
           |                               |- DWI -- 01.jpg 02.jpg ...
           |                               |- DCEE -- 01.jpg 02.jpg ...
           |                               |- DCEL -- 01.jpg 02.jpg ...
           |
           |- val -- patientID -- examID -- TRA -- 01.jpg 02.jpg ...
           |                             |- ADC -- 01.jpg 02.jpg ...
           |                             |- DWI -- 01.jpg 02.jpg ...
           |                             |- DCEE -- 01.jpg 02.jpg ...
           |                             |- DCEL -- 01.jpg 02.jpg ...
           |
           |- test -- patientID -- examID -- TRA -- 01.jpg 02.jpg ...
                                          |- ADC -- 01.jpg 02.jpg ...
                                          |- DWI -- 01.jpg 02.jpg ...
                                          |- DCEE -- 01.jpg 02.jpg ...
                                          |- DCEL -- 01.jpg 02.jpg ...
'''

# built in
import os

# external
import tensorflow as tf

# customs


def train_ds(path, batch_size, buffer_size, repeat=True, slice_types=('TRA', 'ADC', 'DWI', 'DCEE', 'DCEL', 'label')):
    '''
    generate dataset for training

    Args:
        path: train data path
        batch_size: batch size
        buffer_size: buffer_size
        repeat: should ds be repeated
    '''
    ds = base(path, slice_types=slice_types)
    ds = augment(ds)
    ds = to_feature_label(ds, slice_types=slice_types)
    ds = ds.shuffle(buffer_size)
    if repeat: ds = ds.repeat(None)
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

def base(path, slice_types):
    '''
    generate base dataset
    '''
    if os.path.splitext(path) == '.tfrecords':
        return base_from_tfrecords()
    assert os.path.isdir(path)

    pattern = os.path.join(path, *'*' * 2)
    ds = tf.data.Dataset.list_files(pattern)
    ds = ds.map(parse_exam, tf.data.experimental.AUTOTUNE)
    return ds

def parse_exam(exam_dir, slice_types, decoder=tf.image.decode_image):
    '''
    parse exam directory and return contents in dict

    Args:
        exam_dir: exam path
        slice_types: list of slice types to consider
        decoder: image decoder

    Returns:
        dict: {
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
    result['patientID'], result['examID'] = getID_from_exam_path(exam_dir)

    slices_per_type = {
        slice_type: set(os.listdir(os.path.join(exam_dir, slice_types)))
        for slice_type in slice_types
    }
    common_slices = set.intersection(*map(
        lambda slices: set(map(lambda name: int(os.path.splitext(name)[0]), slices)),
        slices_per_type.values(),
    ))
    assert common_slices, f'Not enough slices in {exam_dir}'

    result['nslices'] = len(common_slices)
    for slice_type, names in slice_types.items():
        result[slice_type] = {int(os.path.splitext(name)[0]): decoder(os.path.join(exam_dir, name)) for name in names}
    return result

def getID_from_exam_path(exam_path):
    '''
    parse patientID and examID from given exam_path
    '''
    patient_id, exam_id = map(int, os.path.normpath(exam_path).strip(os.path.sep).split(os.path.sep)[-2:])
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
    feature_slice_types = tuple(e for e in slice_types if e != 'label')

    def convert(dict_):
        assert dict_['nslices'] > 0, f'Found invalid record in ds: {dict_}'
        if dict_['nslices'] > 1: raise NotImplementedError

        label = dict_['label']
        feature = tf.concat(
            list(map(dict_.get, feature_slice_types)),
            axis=2,
        )
        return feature, label

    ds = ds.map(convert, tf.data.experimental.AUTOTUNE)
    return ds
