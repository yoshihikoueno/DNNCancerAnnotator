'''
provides abstraction of DNN library
'''

# built-in
import copy
import os
import pdb
from collections import OrderedDict
from functools import partial
import logging
import re
import itertools

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
        self.model_config = copy.deepcopy(model_config)
        self.model = self.from_config(model_config)
        self.current_step = 0
        self.ckpt_pattern = 'ckpt-{epoch}'
        return

    def get_ckpts(self, base_path):
        regex_pattern = fr'^{self.ckpt_pattern}\.index$'.format(epoch=r'(\d+)')
        files = os.listdir(base_path)
        ckpts_files = list(filter(
            lambda x: re.match(regex_pattern, x),
            files,
        ))
        ckpt_steps = list(map(lambda x: int(re.sub(regex_pattern, r'\1', x)), ckpts_files))
        ckpts_paths = list(map(lambda x: os.path.join(base_path, x[:-len('.index')]), ckpts_files))
        ckpts = dict(zip(ckpt_steps, ckpts_paths))
        return ckpts

    def _auto_resume(self, base_path):
        if not os.path.exists(base_path): return
        ckpts = self.get_ckpts(base_path)
        if not ckpts: return
        latest_step = max(ckpts.keys())
        latest_ckpt = ckpts[latest_step]
        self.model.load_weights(latest_ckpt).assert_consumed()
        self.current_step = latest_step
        logging.warn(f'Resumed from {latest_step}')
        return

    def train(
        self,
        dataset: tf.data.Dataset,
        val_data=None,
        save_path=None,
        save_freq=100,
        max_steps=None,
        early_stop_steps=None,
        visualization=None,
        auto_resume=True,
    ):
        self.model.build(dataset.element_spec[0].shape)
        if auto_resume: self._auto_resume(os.path.join(save_path, 'checkpoints'))
        if visualization is None: visualization = dict()
        callbacks = []
        if save_path is not None:
            ckpt_path = os.path.join(save_path, 'checkpoints', self.ckpt_pattern)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            ckpt_saver = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_freq=save_freq, save_weights_only=True)
            callbacks.append(ckpt_saver)

            tfevents_path = os.path.join(save_path, 'tfevents')
            tb_callback = tf.keras.callbacks.TensorBoard(tfevents_path, update_freq='epoch')
            tb_callback.set_model(self.model)
            callbacks.append(tb_callback)
            for tag, viz_ds in visualization.items():
                viz_callback = custom_callbacks.Visualizer(tag, viz_ds, save_freq, tfevents_path)

                # if tb_callback already have a writer for it, reuse it
                try: viz_callback.writer = getattr(tb_callback, f'_{tag}_writer')
                except AttributeError: pass
                callbacks.append(viz_callback)

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
            initial_epoch=self.current_step,
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
        self.model.load_weights(path)
        return self

    def _saving_hook(self):
        return

    def get_config(self):
        return self.model_config

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

        # workaround to fix opimizer bug in tensorflow
        if deploy_options['optimizer'] == 'adam':
            deploy_options['optimizer'] = tf.keras.optimizers.Adam(
                learning_rate=tf.Variable(0.001),
                beta_1=tf.Variable(0.9),
                beta_2=tf.Variable(0.999),
                epsilon=tf.Variable(1e-7),
            )
            deploy_options['optimizer'].iterations
            deploy_options['optimizer'].decay = tf.Variable(0.0)

        model.compile(**deploy_options)
        return model
