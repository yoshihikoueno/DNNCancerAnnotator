import functools

import tensorflow as tf

from models import unet


def train(input_fn, pipeline_config, result_dir, resume):
  model_fn = functools.partial(unet.estimator_fn,
                               pipeline_config=pipeline_config)

  estimator = tf.estimator.Estimator(model_fn=model_fn)

  estimator.train(input_fn=input_fn,
                  steps=pipeline_config.train_config.num_steps)
