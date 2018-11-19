from google.protobuf import text_format

from protos import pipeline_pb2
from builders import dataset_builder
from utils import standard_fields


def load_config(pipeline_config_path):
  pipeline_config = pipeline_pb2.PipelineConfig()
  with open(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  return pipeline_config


def get_input_fn(pipeline_config, directory, existing_tfrecords_directory,
                 split_name):
  target_dims = [pipeline_config.model.input_image_size_x,
                 pipeline_config.model.input_image_size_y,
                 pipeline_config.model.input_image_channels]

  if split_name == standard_fields.SplitNames.train:
    batch_size = pipeline_config.train_config.batch_size
  else:
    batch_size = pipeline_config.eval_config.batch_size

  def input_fn():
    return dataset_builder.build_dataset(
      pipeline_config.dataset, directory=directory,
      existing_tfrecords_directory=existing_tfrecords_directory,
      split_name=split_name, target_dims=target_dims,
      seed=pipeline_config.seed, batch_size=batch_size,
      shuffle=pipeline_config.train_config.shuffle,
      shuffle_buffer_size=pipeline_config.train_config.shuffle_buffer_size)

  return input_fn
