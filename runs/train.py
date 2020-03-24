'''
interface for training models
'''

# built-in
import pdb
import os
import argparse

# external

# customs
import engine
import data
from utils import dump
from utils import load


def main(
    model_config,
    save_path,
    data_path,
):
    '''
    Train a model with specified configs.
    This funciton will first dump the input arguments,
    then train a model, finally dump reults.

    Args:
        model_config (dict): model configuration
        save_path: where to save weights/configs/results
        data_path: path to the data root dir
    '''
    dump.dump_options(
        os.path.join(save_path, 'options.json'),
        format_='json',
        model_config=model_config,
        save_path=save_path,
        data_path=data_path,
    )

    ds = data.train_ds(data_path)
    model_config = load.load_model_config(model_config)
    model = engine.TFKerasModel(model_config, save_path)
    results = model.train(ds)

    dump.dump_train_results(
        os.path.join(save_path, 'resutls.pickle'),
        results,
    )
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_config',
        help='Path to the model config file.',
    )
    parser.add_argument(
        '--save_path',
        help='Path to the directory where model data are saved in and loaded from.',
    )
    command = parser.add_subparsers(help='Action to take.', metavar='command', dest='command')

    train_parser = command.add_parser('train', help='Train a model')
    eval_parser = command.add_parser('eval', help='Train a model')
    predict_parser = command.add_parser('predict', help='Train a model')

    args = parser.parse_args()
    main(**vars(args))
