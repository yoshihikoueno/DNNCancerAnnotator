'''
provide custom callbacks
'''

# biult-in
import pdb
import os
import sys
import pprint
from multiprocessing import cpu_count

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
        self.progbar.update(1)
        return

    def on_train_end(self, logs={}):
        tqdm.write(pprint.pformat(logs))
        if self.progbar is None: return
        self.progbar.close()
        return


class Visualizer(Callback):
    def __init__(self, tag: str, data: tf.data.Dataset, freq: int, save_dir: str, ratio=0.5):
        self.params = None
        self.model = None
        self.tag = tag
        self.data = data
        self.data_size = None
        self.freq = freq
        self.save_dir = save_dir
        self.ratio = ratio
        self.writer = None
        super().__init__()
        self.set_data_size()
        return

    def set_data_size(self):
        count = 0
        for _ in self.data: count += 1
        self.data_size = count
        return

    def on_epoch_end(self, *args):
        if self.get_current_step() % self.freq != 0: return
        if self.writer is None: self.writer = tf.summary.create_file_writer(os.path.join(self.save_dir, self.tag))
        with self.writer.as_default():
            list(map(self.process_batch, tqdm(self.data, desc='visualizing', total=self.data_size)))
        self.writer.flush()
        return

    def on_train_end(self, *args):
        if self.writer is None: return
        self.writer.close()
        self.writer = None
        return

    def process_batch(self, batch):
        consts = self.make_summary_batch(batch)
        for tag, image, step in zip(*consts):
            result = tf.summary.image(tag.numpy().decode(), image, step=step)
            assert result
        return

    @tf.function
    def make_summary_batch(self, batch):
        batch_output = self.model(batch['x'])
        results = tf.map_fn(
            lambda x: self.make_summary_constructor(x[0], x[1], x[2], x[3]),
            (batch['x'], batch['y'], batch['path'], batch_output),
            dtype=(tf.string, tf.float32, tf.int64),
            parallel_iterations=cpu_count(),
        )
        return results

    @tf.function
    def make_summary_constructor(self, features, label, path, output):
        image = self.generate_image(features, label, output)
        image = tf.image.resize(image, tf.cast(tf.cast(tf.shape(image)[1:3], tf.float32) * self.ratio, tf.int32))
        return tf.strings.join(['path', path]), image, self.get_current_step()

    def make_summary(self, features, label, path, output):
        image = self.generate_image(features, label, output)
        image = tf.image.resize(image, tf.cast(tf.cast(tf.shape(image)[1:3], tf.float32) * self.ratio, tf.int32))
        tf.summary.image('path' + path.numpy().decode(), image, step=self.get_current_step())
        return

    def get_current_step(self):
        step = self.model._train_counter
        return step

    def generate_image(self, features, label, output, axis=1):
        assert len(features.shape) == 3
        horizontal_features = tf.concat(tf.unstack(features, axis=-1), axis=axis)
        image = tf.concat([horizontal_features, label, tf.squeeze(output, axis=-1)], axis=axis)
        image = tf.expand_dims(image, 0)
        image = tf.expand_dims(image, -1)
        return image
