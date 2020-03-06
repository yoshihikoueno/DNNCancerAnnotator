'''
provides abstraction of DNN library
'''

# built-in
import copy

# external
import tensorflow as tf
from tensorflow import keras

# customs
import tf_models


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
        self.model_package = tf_models
        self.model_fn = self._model_config_to_model_fn(model_config)
        self.model = keras.Model(self.model_fn)
        return

    def train(self, dataset):
        results = self.model.fit()
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

    def _model_config_to_model_fn(self, model_config):
        model_name = model_config['name']
        model_config = copy.deepcopy(model_config)
        del model_config['name']

        model_fn = self.model_package[model_name](**model_config)
        return model_fn
