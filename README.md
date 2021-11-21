# Predict cancer segmentations using DNN

## Framework
Tensorflow

## Models
 * U-Net
 * MulmoU-Net

## Install
 ```bash
 pip3 install git+https://github.com/yoshihikoueno/DNNCancerAnnotator.git@master
 ```

## Usage
### Available commands
 ```bash
 python3 -m annotator -h

 # usage: python3 -m annotator [-h] {train,evaluate,extract_all,generate_tfrecords} ...
 # 
 # DNNAnnotator: CLI interface
 # 
 # positional arguments:
 #   {train,evaluate,extract_all,generate_tfrecords}
 #                         command
 #     train               Train a model with specified configs.
 #     evaluate            Evaluate a model with specified configs
 #     extract_all         extract indivisual images (TRA, ADC, etc...) from the screenshots
 #     generate_tfrecords  Generate TFRecords
 # 
 # optional arguments:
 #   -h, --help            show this help message and exit
 ```
 

### Train
 Command:
 ```bash
 python3 -m annotator train
 ```

 Options:
 ```bash
 python3 -m annotator train -h

 # usage: python3 -m annotator train [-h] --config CONFIG [CONFIG ...] --save_path SAVE_PATH --data_path DATA_PATH [DATA_PATH ...]
 #                                   --max_steps MAX_STEPS [--early_stop_steps EARLY_STOP_STEPS] [--save_freq SAVE_FREQ] [--validate]
 #                                   [--val_data_path VAL_DATA_PATH [VAL_DATA_PATH ...]] [--visualize] [--profile]
 # 
 # Train a model with specified configs.
 # 
 # This function will first dump the input arguments,
 # then train a model, finally dump reults.
 # 
 # optional arguments:
 #   -h, --help            show this help message and exit
 #   --config CONFIG [CONFIG ...]
 #                         configuration file path
 #                             This option accepts arbitrary number of configs.
 #                             If a list is specified, the first one is considered
 #                             as a "main" config, and the other ones will overwrite the content
 #   --save_path SAVE_PATH
 #                         where to save weights/configs/results
 #   --data_path DATA_PATH [DATA_PATH ...]
 #                         path to the data root dir
 #   --max_steps MAX_STEPS
 #                         max training steps
 #   --early_stop_steps EARLY_STOP_STEPS
 #                         steps to train without improvements
 #                             None(default) disables this feature
 #   --save_freq SAVE_FREQ
 #                         interval of checkpoints
 #                             default: 500 steps
 #   --validate            also validate the model on the validation dataset
 #   --val_data_path VAL_DATA_PATH [VAL_DATA_PATH ...]
 #                         path to the validation dataset
 #   --visualize           should visualize results
 #   --profile             enable profilling
 ```
 

### Evaluate
 Command:
 ```bash
 python3 -m annotator evaluate
 ```

 Options:
 ```bash
 python3 -m annotator evaluate -h

 # usage: python3 -m annotator evaluate [-h] --save_path SAVE_PATH --data_path DATA_PATH [DATA_PATH ...] --tag TAG [--config CONFIG]
 #                                      [--avoid_overwrite] [--export_path EXPORT_PATH] [--export_images] [--export_csv]
 #                                      [--min_interval MIN_INTERVAL]
 # 
 # Evaluate a model with specified configs
 # 
 # for every checkpoints available.
 # 
 # optional arguments:
 #   -h, --help            show this help message and exit
 #   --save_path SAVE_PATH
 #                         where to find weights/configs/results
 #   --data_path DATA_PATH [DATA_PATH ...]
 #                         path to the data root dir
 #   --tag TAG             save tag
 #   --config CONFIG       configuration file path
 #                             None (default): load config from save_path
 #   --avoid_overwrite     should `save_path` altered when a directory already
 #                             exists at the original `save_path` to avoid overwriting.
 #   --export_path EXPORT_PATH
 #                         path to export results
 #   --export_images       export images
 #   --export_csv          export results csv
 #   --min_interval MIN_INTERVAL
 #                         minimum interval in steps between evaluations.
 #                             Checkpoints which are less than `min_interval` steps away
 #                             from the previous one will be disregarded.
 ```
