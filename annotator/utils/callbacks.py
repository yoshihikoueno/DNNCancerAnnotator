'''
provide custom callbacks
'''

# biult-in
import pdb
import os
import sys
import pprint
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

# external
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback


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
    def __init__(self, tag: str, data: tf.data.Dataset, freq: int, save_dir: str, ratio=0.5, prediction_threshold=None):
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
        self.prediction_threshold = prediction_threshold
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
        self.set_current_step(epoch)
        if self.get_current_step() % self.freq != 0: return
        self.record_visuals()
        return

    def record_visuals(self):
        with self.writer.as_default():
            list(map(self.process_batch, tqdm(self.data, desc='visualizing', total=self.data_size)))
        self.writer.flush()
        return

    def __del__(self, *args, **kargs):
        if self._writer is not None and self._owned_writer:
            self._writer.close()
            self._writer = None
        return

    def record_logs(self, logs):
        epoch = self.get_current_step()
        with self.writer.as_default():
            for summary_name, value in logs.items():
                tf.summary.scalar(summary_name, value, epoch)
        self.writer.flush()
        return

    def record_all(self, epoch, logs=None):
        if logs is not None: self.record_logs(epoch, logs)
        self.record_visuals(epoch)
        return

    def on_test_end(self, logs={}):
        self.record_visuals()
        self.record_logs(logs)
        return

    def on_train_end(self, *args):
        if self.writer is None: return
        if self._owned_writer: self.writer.close()
        self._writer = None
        self._owned_writer = True
        return

    def process_batch(self, batch):
        consts = self.make_summary_batch(batch)
        with ThreadPoolExecutor() as e:
            results = list(e.map(self._emit, *consts))
        assert tf.reduce_all(results)
        return

    def _emit(self, tag, image):
        with self.writer.as_default():
            result = tf.summary.image(tag.numpy().decode(), image, step=self.current_step)
        return result

    @tf.function
    def make_summary_batch(self, batch):
        batch_output = self.model(batch['x'])
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

    def make_summary(self, features, label, path, output):
        image = self.generate_image(features, label, output)
        image = tf.image.resize(image, tf.cast(tf.cast(tf.shape(image)[1:3], tf.float32) * self.ratio, tf.int32))
        tf.summary.image('path' + path.numpy().decode(), image, step=self.get_current_step())
        return

    def set_current_step(self, step):
        self.current_step = step
        return

    def get_current_step(self):
        step = tf.cast(self.current_step, tf.int64)
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
