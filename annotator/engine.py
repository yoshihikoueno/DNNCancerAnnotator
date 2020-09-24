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
import pandas as pd

# customs
from .models import tf_models
from .utils import losses as custom_losses
from .utils import metrics as custom_metrics
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
        self.model: tf.keras.Model = self.from_config(model_config)
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
        ckpts = OrderedDict(sorted(zip(ckpt_steps, ckpts_paths), key=lambda x: x[0]))
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
        self._enter_strategy_section()
        self.model.build(dataset.element_spec[0].shape)
        if auto_resume: self._auto_resume(os.path.join(save_path, 'checkpoints'))
        if visualization is None: visualization = dict()
        callbacks = []
        if self.learning_rate_scheduler is not None:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(
                eval(self.learning_rate_scheduler), verbose=1,
            ))

        if save_path is not None:
            ckpt_path = os.path.join(save_path, 'checkpoints', self.ckpt_pattern)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            ckpt_saver = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_freq=save_freq, save_weights_only=True)
            callbacks.append(ckpt_saver)

            tfevents_path = os.path.join(save_path, 'tfevents')
            tb_callback = tf.keras.callbacks.TensorBoard(tfevents_path, update_freq='epoch', profile_batch=0)
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
        self._exit_strategy_section()
        return results

    def eval(
            self,
            dataset,
            viz_ds,
            save_path,
            tag='val',
            avoid_overwrite=False,
            export_path=None,
            export_images=False,
            export_csv=False,
    ):
        self.model.build(dataset.element_spec[0].shape)
        ckpt_path = os.path.join(save_path, 'checkpoints')

        if not export_path: export_path = os.path.join(save_path, 'tfevents')
        if os.path.exists(os.path.join(export_path, tag)):
            if avoid_overwrite:
                while os.path.exists(os.path.join(export_path, tag)): tag += '_'
            else: raise ValueError(f'tag: {tag} already exists.')

        viz_callback = custom_callbacks.Visualizer(
            tag, viz_ds, 1, ignore_test=False,
            save_dir=export_path,
            export_images=export_images,
        )
        if export_csv:
            result_container = pd.DataFrame()
            result_container.index.rename('step', inplace=True)
        for ckpt_step, ckpt_path_ in tqdm(self.get_ckpts(ckpt_path).items(), desc='checkpoints'):
            viz_callback.set_current_step(ckpt_step)
            self.load(ckpt_path_)
            results = self.model.evaluate(dataset, callbacks=[viz_callback], verbose=0, return_dict=True)
            if export_csv: result_container = result_container.append(pd.Series(results, name=ckpt_step))
        if export_csv:
            result_container.to_csv(os.path.join(export_path, tag, 'results.csv'))
        return

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

    def _enter_strategy_section(self):
        assert getattr(self, '_scope', None) is None
        if not self.enable_multigpu:
            self._scope = None
            return
        self._scope = self.strategy.scope()
        self._scope.__enter__()
        return

    def _exit_strategy_section(self):
        if getattr(self, '_scope', None) is None: return
        self._scope.__exit__(None, None, None)
        del self._scope
        return

    def from_config(self, model_config) -> tf.keras.Model:
        assert 'model' in model_config
        assert 'model_options' in model_config
        assert 'deploy_options' in model_config

        deploy_options = copy.deepcopy(model_config['deploy_options'])
        self.enable_multigpu = deploy_options.pop('enable_multigpu', True)
        if self.enable_multigpu:
            self.strategy = tf.distribute.MirroredStrategy()
        self._enter_strategy_section()

        self.learning_rate_scheduler = deploy_options.pop('LearningRateScheduler', None)

        model_name = model_config['model']
        model = getattr(tf_models, model_name)(**model_config['model_options'])

        if 'loss' in deploy_options:
            deploy_options['loss'] = tf.keras.losses.get(deploy_options['loss'])

        deploy_options['metrics'] = list(map(custom_metrics.solve_metric, deploy_options.get('metrics', [])))

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
        self._exit_strategy_section()
        return model
