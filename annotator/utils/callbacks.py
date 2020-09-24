'''
provide custom callbacks
'''

# biult-in
import pdb
import os
import pprint
import re
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

# external
import tensorflow as tf
from tensorboard import summary as summary_lib
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback

# custom
from . import metrics as custom_metrics


class TFProgress(Callback):
    def __init__(self):
        self.params = None
        self.progbar = None
        super().__init__()
        return

    def on_train_begin(self, logs=None):
        assert self.progbar is None
        if logs is not None: tqdm.write(pprint.pformat(logs))
        self.progbar = tqdm(total=self.params['epochs'])
        return

    def on_epoch_end(self, epoch, logs={}):
        tqdm.write(f'Epoch:{epoch}\n' + pprint.pformat(logs))
        if self.progbar is None: return
        self.progbar.n = epoch + 1
        self.progbar.last_print_n = epoch + 1
        return

    def on_train_end(self, logs={}):
        tqdm.write(pprint.pformat(logs))
        if self.progbar is None: return
        self.progbar.close()
        return


class Visualizer(Callback):
    '''visualizer callback

    This callback will export summaries including:
        - vizualization of segmentations
        - metrics (only during testing)
        - PR curve

    Args:
        tag: summary tag
        data: dataset to be used for vizualization and PR curve
            samples are drawn from this dataset and fed into model
        freq: save frequency
        save_dir: summary save directory
            This will be ignored if writer is to be registered.
        ratio: dimension ratio of exported vizualization images.
            For example, if the samples in dataset has size (256, 256)
            and ratio is 0.5, then the output images will have size (128, 128).
        prediction_threshold: threshold to apply against predicted segmentation
        pr_nthreshold: the number of thresholds to use for PR curve
        pr_region_nthreshold: the number of thresholds to use for region based PR curve
        ignore_test: whether this callback should do nothing for test events
    '''
    def __init__(
        self,
        tag: str,
        data: tf.data.Dataset,
        freq: int,
        save_dir: str,
        ratio=0.5,
        prediction_threshold=None,
        pr_nthreshold=100,
        pr_region_nthreshold=100,
        pr_IoU_threshold=0.30,
        ignore_test=True,
        export_images=False,
        export_path_depth=3,
    ):
        self.params = None
        self.model = None
        self.tag = tag
        self.data = data
        self.data_size = None
        self.freq = freq
        self.save_dir = save_dir
        self.ratio = ratio
        self._writer = None
        self._owned_writer = True
        self.export_images = export_images
        self.export_path_depth = export_path_depth
        self.pr_nthreshold = pr_nthreshold
        self.pr_region_nthreshold = pr_region_nthreshold
        self.pr_IoU_threshold = pr_IoU_threshold
        self.ignore_test = ignore_test
        self.per_epoch_resources = {}
        self.prediction_threshold = prediction_threshold
        self.internal_metics = [
            'true_positive_counts', 'true_negative_counts',
            'false_positive_counts', 'false_negative_counts',
            'recall', 'precision',
        ]
        super().__init__()
        self.set_data_size()
        return

    @property
    def writer(self):
        if self._writer is None:
            self._writer = tf.summary.create_file_writer(os.path.join(self.save_dir, self.tag))
            self._owned_writer = True
        return self._writer

    @writer.setter
    def writer(self, writer):
        if self._writer is not None and self._owned_writer:
            self._writer.close()
            self._writer = None
        self._writer = writer
        self._owned_writer = False
        return

    def set_data_size(self):
        count = 0
        for _ in self.data: count += 1
        self.data_size = count
        return

    def on_epoch_end(self, epoch, logs=None):
        self.prepare_internal_metrics()
        self.set_current_step(epoch)
        if self.get_current_step() % self.freq != 0: return
        self.record_visuals()
        self.record_pr_curve()
        self.per_epoch_resources = {}
        return

    def record_visuals(self):
        with self.writer.as_default():
            list(map(self.process_batch, tqdm(self.data, desc='visualizing', total=self.data_size)))
        self.writer.flush()
        return

    def record_pr_curve(self):
        data = self.get_internal_metrics_results()
        thresholds = {
            'pixel': self.pr_nthreshold,
            'region': self.pr_region_nthreshold,
        }
        for type_, data_ in data.items():
            summary_pb = summary_lib.v1.pr_curve_raw_data_pb(
                **data_,
                num_thresholds=thresholds[type_],
                name=f'{type_}/PR_curve',
            )
            with self.writer.as_default():
                tf.summary.experimental.write_raw_pb(summary_pb.SerializeToString(), step=self.get_current_step())
        return

    def __del__(self, *args, **kargs):
        if self._writer is not None and self._owned_writer:
            self._writer.close()
            self._writer = None
        return

    def prepare_internal_metrics(self):
        pixel_thresholds = [i / float(self.pr_nthreshold - 1) for i in range(self.pr_nthreshold)]
        region_thresholds = [i / float(self.pr_region_nthreshold - 1) for i in range(self.pr_region_nthreshold)]
        self.per_epoch_resources['pr_curve'] = {
            'pixel': {
                'true_positive_counts': tf.keras.metrics.TruePositives(pixel_thresholds),
                'true_negative_counts': tf.keras.metrics.TrueNegatives(pixel_thresholds),
                'false_positive_counts': tf.keras.metrics.FalsePositives(pixel_thresholds),
                'false_negative_counts': tf.keras.metrics.FalseNegatives(pixel_thresholds),
                'recall': tf.keras.metrics.Recall(pixel_thresholds),
                'precision': tf.keras.metrics.Precision(pixel_thresholds),
            },
            'region': custom_metrics.RegionBasedConfusionMatrix(
                region_thresholds,
                self.pr_IoU_threshold,
                resize_factor=self.ratio,
            ),
        }
        return

    def update_internal_metrics(self, y_true, y_pred):
        for type_ in self.per_epoch_resources['pr_curve']:
            if type_ == 'pixel':
                for metric in self.internal_metics:
                    self.per_epoch_resources['pr_curve'][type_][metric].update_state(y_true, y_pred)
            elif type_ == 'region':
                self.per_epoch_resources['pr_curve'][type_].update_state(y_true, y_pred)
            else: raise NotImplementedError
        return

    def get_internal_metrics_results(self):
        result = dict()
        for type_ in self.per_epoch_resources['pr_curve']:
            if type_ == 'pixel':
                result[type_] = {
                    metric: self.per_epoch_resources['pr_curve'][type_][metric].result()
                    for metric in self.internal_metics
                }
            elif type_ == 'region':
                result[type_] = dict(
                    true_negative_counts=[0] * self.pr_region_nthreshold,
                    **self.per_epoch_resources['pr_curve'][type_].result_dict(),
                )
            else: raise NotImplementedError
        return result

    def record_logs(self, logs):
        epoch = self.get_current_step()
        with self.writer.as_default():
            for summary_name, value in logs.items():
                tf.summary.scalar(summary_name, value, epoch)
        self.writer.flush()
        return

    def on_test_end(self, logs={}):
        if self.ignore_test: return
        self.prepare_internal_metrics()
        self.record_visuals()
        self.record_pr_curve()
        self.record_logs(logs)
        self.per_epoch_resources = {}
        return

    def on_train_end(self, *args):
        if self.writer is None: return
        if self._owned_writer: self.writer.close()
        self._writer = None
        self._owned_writer = True
        return

    def process_batch(self, batch):
        batch_output = self.model(batch['x'])
        self.update_internal_metrics(batch['y'], batch_output)
        consts = self.make_summary_batch(batch, batch_output)
        with ThreadPoolExecutor() as e:
            results = list(e.map(self._emit, *consts))
        assert tf.reduce_all(results)
        return

    def _emit(self, tag, image):
        with self.writer.as_default():
            result = tf.summary.image(tag.numpy().decode(), image, step=self.get_current_step())
        if self.export_images:
            tag_str = tag.numpy().decode()
            pattern = r'^path:(.*),sliceID:(.*)$'
            tags = re.sub(pattern, r'\1', tag_str).split('/')[-self.export_path_depth:]
            slice_num = int(re.sub(pattern, r'\2', tag_str))
            step = self.get_current_step()
            path = os.path.join(self.save_dir, self.tag, 'images', *tags, f'{slice_num:02d}', f'step_{step:08d}.png')
            tf.io.write_file(path, tf.image.encode_png(tf.squeeze(tf.cast(image * 255, tf.uint8), 0)))
        return result

    @tf.function
    def make_summary_batch(self, batch, batch_output):
        results = tf.map_fn(
            lambda x: self.make_summary_constructor(x[0], x[1], x[2], x[3], x[4]),
            (batch['x'], batch['y'], batch['path'], batch['sliceID'], batch_output),
            dtype=(tf.string, tf.float32),
            parallel_iterations=cpu_count(),
        )
        return results

    @tf.function
    def make_summary_constructor(self, features, label, path, sliceID, output):
        image = self.generate_image(features, label, output)
        image = tf.image.resize(image, tf.cast(tf.cast(tf.shape(image)[1:3], tf.float32) * self.ratio, tf.int32))
        sliceID = tf.strings.as_string(sliceID)
        return tf.strings.join(['path:', path, ',sliceID:', sliceID]), image

    def set_current_step(self, step):
        self.per_epoch_resources['step'] = step
        return

    def get_current_step(self):
        step = self.per_epoch_resources['step']
        step = tf.cast(step, tf.int64)
        return step

    def generate_image(self, features, label, output, axis=1):
        assert len(features.shape) == 3
        horizontal_features = tf.concat(tf.unstack(features, axis=-1), axis=axis)
        pred = tf.squeeze(output, axis=-1)
        if self.prediction_threshold is not None: pred = tf.cast(pred > self.prediction_threshold, pred.dtype)
        image = tf.concat([horizontal_features, label, pred], axis=axis)
        image = tf.expand_dims(image, 0)
        image = tf.expand_dims(image, -1)
        return image
