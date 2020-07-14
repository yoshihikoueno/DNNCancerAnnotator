'''
provides abstraction of DNN library
'''

# built-in
import copy
import os
import pdb
from collections import OrderedDict
from functools import partial

# external
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# customs
from .models import tf_models
from .utils import losses as custom_losses
from .utils import callbacks as custom_callbacks

def _set_model(self, model):
    self.model = model
    return


tf.keras.callbacks.ModelCheckpoint.set_model = _set_model


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

    def train(
        self,
        dataset,
        val_data=None,
        save_path=None,
        save_freq=100,
        max_steps=None,
        early_stop_steps=None,
        visualization=None,
    ):
        if visualization is None: visualization = dict()
        callbacks = []
        if save_path is not None:
            ckpt_path = os.path.join(save_path, 'checkpoints', 'ckpt-{epoch}')
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            ckpt_saver = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_freq=save_freq)
            callbacks.append(ckpt_saver)

            tfevents_path = os.path.join(save_path, 'tfevents')
            callbacks.append(tf.keras.callbacks.TensorBoard(tfevents_path, update_freq=save_freq))
            for tag, viz_ds in visualization.items():
                callbacks.append(custom_callbacks.Visualizer(tag, viz_ds, save_freq, tfevents_path))

        if early_stop_steps is not None:
            stopper = tf.keras.callbacks.EarlyStopping(patience=early_stop_steps, verbose=1)
            callbacks.append(stopper)

        callbacks.append(custom_callbacks.TFProgress())

        results = self.model.fit(
            dataset,
            validation_data=val_data,
            callbacks=callbacks,
            steps_per_epoch=1,
            epochs=max_steps,
            validation_freq=save_freq,
            verbose=0,
        )
        return results

    def eval(self, dataset, save_path, ckpt_path, step=None, daemon=False):
        if daemon: raise NotImplementedError
        assert not os.path.exists(save_path)
        assert os.path.exists(ckpt_path)

        ckpts = self.list_ckpts(ckpt_path)
        if step is None:
            for step_, ckpt_path_ in tqdm(ckpts):
                self.eval(dataset, save_path, ckpt_path_, step=step_, daemon=False)
            return
        else:
            results = self.model.evaluate(dataset)
            return results

    def list_ckpts(self, save_path):
        assert os.path.exists(save_path)
        files = os.listdir(save_path)
        files = list(filter(lambda x: x.startswith('ckpt-'), files))
        id_ckptpath = list(map(lambda x: (int(x[5:]), os.path.join(save_path, x)), files))
        id_ckptpath = sorted(id_ckptpath, key=lambda x: x[0])
        ckpts = OrderedDict(id_ckptpath)
        return ckpts

    def predict(self, dataset):
        results = self.model.predict(dataset)
        return results

    def save(self, path, fileformat):
        self.model.save(path)
        return self

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        return self

    def _saving_hook(self):
        return

    def from_config(self, model_config):
        assert 'model' in model_config
        assert 'model_options' in model_config
        assert 'deploy_options' in model_config

        deploy_options = copy.deepcopy(model_config['deploy_options'])
        enable_multigpu = deploy_options.pop('enable_multigpu', True)

        model_name = model_config['model']
        if enable_multigpu:
            with tf.distribute.MirroredStrategy().scope():
                model = getattr(tf_models, model_name)(**model_config['model_options'])
        else:
            model = getattr(tf_models, model_name)(**model_config['model_options'])

        if isinstance(deploy_options.get('loss', None), dict):
            loss_config = deploy_options.pop('loss')
            loss_name = loss_config['name']
            loss_option = loss_config.get('option', {})
            loss_class = tf.keras.utils.get_registered_object(loss_name)
            loss = loss_class(**loss_option)
            deploy_options['loss'] = loss

        model.compile(**deploy_options)
        return model
