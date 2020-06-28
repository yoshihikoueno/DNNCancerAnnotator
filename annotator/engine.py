'''
provides abstraction of DNN library
'''

# built-in
import copy
import os
import pdb

# external
import tensorflow as tf
from tensorflow import keras

# customs
from .models import tf_models


class TFKerasModel():
    '''
    this class emcapsulates DNN model
    and libraries behind it.

    Attributes:
        model_package: provides mapping of model_name to model_fn
    '''
    def __init__(self, model_config):
        '''
        Args:
            model_config (dict):m model configuration
        '''
        self.model = self.from_config(model_config)
        return

    def train(self, dataset, save_path=None, save_freq=100, max_steps=None, early_stop_steps=None):
        callbacks = []
        if save_path is not None:
            ckpt_path = os.path.join(save_path, 'checkpoints', 'ckpt-{epoch}.hdf5')
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_freq=save_freq))

            tfevents_path = os.path.join(save_path, 'tfevents')
            callbacks.append(tf.keras.callbacks.TensorBoard(tfevents_path, update_freq=save_freq))

        if early_stop_steps is not None:
            stopper = tf.keras.callbacks.EarlyStopping(patience=early_stop_steps, verbose=1)
            callbacks.append(stopper)

        results = self.model.fit(dataset, callbacks=callbacks, steps_per_epoch=1, epochs=max_steps)
        return results

    def eval(self, dataset):
        results = self.model.evaluate(dataset)
        return results

    def predict(self, dataset):
        results = self.model.predict(dataset)
        return results

    def save(self, path, fileformat):
        self.model.save(path)
        return self

    def load(self, path, fileformat):
        self.model.load(path)
        return self

    def _saving_hook(self):
        return

    def from_config(self, model_config):
        assert 'model' in model_config
        assert 'model_options' in model_config
        assert 'deploy_options' in model_config

        model_name = model_config['model']
        with tf.distribute.MirroredStrategy().scope():
            model = getattr(tf_models, model_name)(**model_config['model_options'])
        model.compile(**model_config['deploy_options'])
        return model
