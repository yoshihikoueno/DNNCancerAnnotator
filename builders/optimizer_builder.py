import tensorflow as tf


def build(optimizer_config):
  optimizer_type = optimizer_config.WhichOneof('optimizer')

  lr = _create_learning_rate(optimizer_config.learning_rate)

  summary_vars = [lr]
  if optimizer_type == 'rms_prop_optimizer':
    optimizer = tf.train.RMSPropOptimizer(
      lr,
      decay=optimizer_config.rms_prop_optimizer.decay,
      momentum=optimizer_config.rms_prop_optimizer.momentum_optimizer_value,
      epsilon=optimizer_config.rms_prop_optimizer.epsilon)
  elif optimizer_type == 'momentum_optimizer':
    optimizer = tf.train.MomentumOptimizer(
      lr,
      momentum=optimizer_config.momentum_optimizer.momentum_optimizer_value)
  elif optimizer_type == 'adam_optimizer':
    optimizer = tf.train.AdamOptimizer(
      lr,
      epsilon=optimizer_config.adam_optimizer.epsilon)
  else:
    raise ValueError('Optimizer {} not supported.'.format(optimizer_type))

  if optimizer_config.use_moving_average:
    optimizer = tf.contrib.opt.MovingAverageOptimizer(
      optimizer, average_decay=optimizer_config.moving_average_decay)

  return optimizer, summary_vars


def _create_learning_rate(learning_rate_config):
  """Create optimizer learning rate based on config.
  Args:
    learning_rate_config: A LearningRate proto message.
  Returns:
    A learning rate.
  Raises:
    ValueError: when using an unsupported input data type.
  """
  learning_rate = None
  learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
  if learning_rate_type == 'constant_learning_rate':
    config = learning_rate_config.constant_learning_rate
    learning_rate = tf.constant(config.learning_rate, dtype=tf.float32)

  if learning_rate_type == 'exponential_decay_learning_rate':
    config = learning_rate_config.exponential_decay_learning_rate
    learning_rate = tf.train.exponential_decay(
        config.initial_learning_rate,
        tf.train.get_or_create_global_step(),
        config.decay_steps,
        config.decay_factor,
        staircase=config.staircase)

  if learning_rate_type == 'manual_step_learning_rate':
    config = learning_rate_config.manual_step_learning_rate
    if not config.schedule:
      raise ValueError('Empty learning rate schedule.')
    learning_rate_step_boundaries = [x.step for x in config.schedule]
    learning_rate_sequence = [config.initial_learning_rate]
    learning_rate_sequence += [x.learning_rate for x in config.schedule]
    learning_rate = _manual_stepping(
        tf.train.get_or_create_global_step(), learning_rate_step_boundaries,
        learning_rate_sequence)

  if learning_rate is None:
    raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

  return learning_rate


def _manual_stepping(global_step, boundaries, rates):
  """Manually stepped learning rate schedule.
  This function provides fine grained control over learning rates.  One must
  specify a sequence of learning rates as well as a set of integer steps
  at which the current learning rate must transition to the next.  For example,
  if boundaries = [5, 10] and rates = [.1, .01, .001], then the learning
  rate returned by this function is .1 for global_step=0,...,4, .01 for
  global_step=5...9, and .001 for global_step=10 and onward.
  Args:
    global_step: int64 (scalar) tensor representing global step.
    boundaries: a list of global steps at which to switch learning
      rates.  This list is assumed to consist of increasing positive integers.
    rates: a list of (float) learning rates corresponding to intervals between
      the boundaries.  The length of this list must be exactly
      len(boundaries) + 1.
  Returns:
    a (scalar) float tensor representing learning rate
  Raises:
    ValueError: if one of the following checks fails:
      1. boundaries is a strictly increasing list of positive integers
      2. len(rates) == len(boundaries) + 1
  """
  if any([b < 0 for b in boundaries]) or any(
      [not isinstance(b, int) for b in boundaries]):
    raise ValueError('boundaries must be a list of positive integers')
  if any([bnext <= b for bnext, b in zip(boundaries[1:], boundaries[:-1])]):
    raise ValueError('Entries in boundaries must be strictly increasing.')
  if any([not isinstance(r, float) for r in rates]):
    raise ValueError('Learning rates must be floats')
  if len(rates) != len(boundaries) + 1:
    raise ValueError('Number of provided learning rates must exceed '
                     'number of boundary points by exactly 1.')
  if not boundaries:
    return tf.constant(rates[0])
  step_boundaries = tf.constant(boundaries, tf.int32)
  num_boundaries = len(boundaries)
  learning_rates = tf.constant(rates, tf.float32)
  index = tf.reduce_min(
      tf.where(
          # Casting global step to tf.int32 is dangerous, but necessary to be
          # compatible with TPU.
          tf.greater(step_boundaries, tf.cast(global_step, tf.int32)),
          tf.constant(list(range(num_boundaries)), dtype=tf.int32),
          tf.constant([num_boundaries] * num_boundaries, dtype=tf.int32)))
  return tf.reduce_sum(learning_rates * tf.one_hot(index, len(rates),
                                                   dtype=tf.float32))
