"""
Project: codealpha_ai_task03-
Created: August 2024
Description: Evaluate trained model checkpoints against Nottingham test data.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import tensorflow.compat.v1 as tf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from music_rnn import util
from music_rnn import nottingham_util
from music_rnn.model import NottinghamModel, NottinghamSeparate
from rnn import DefaultConfig

tf.disable_v2_behavior()


def load_config(config_file):
    import __main__

    __main__.DefaultConfig = DefaultConfig
    with open(config_file, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    np.random.seed()      

    parser = argparse.ArgumentParser(description='Script to test a models performance against the test set')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--pickle_path', type=str, default=nottingham_util.PICKLE_LOC)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--separate', action='store_true', default=False)
    parser.add_argument('--seperate', action='store_true', default=False)
    parser.add_argument('--choice', type=str, default='melody',
                        choices = ['melody', 'harmony'])
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        raise FileNotFoundError("Config file not found: {}".format(args.config_file))

    config = load_config(args.config_file)

    separate_mode = args.separate or args.seperate

    if config.dataset == 'softmax':
        config.time_batch_len = 1
        config.max_time_batches = -1
        if not os.path.exists(args.pickle_path):
            raise FileNotFoundError(
                "Dataset pickle not found: {}. Run scripts/create_demo_data.py or scripts/main.py first.".format(
                    args.pickle_path
                )
            )
        with open(args.pickle_path, 'rb') as f:
            data_pickle = pickle.load(f)
        if separate_mode:
            model_class = NottinghamSeparate
            test_data = util.batch_data(data_pickle['test'], time_batch_len = 1, 
                max_time_batches = -1, softmax = True)
            r = nottingham_util.NOTTINGHAM_MELODY_RANGE
            if args.choice == 'melody':
                print("Using only melody")
                new_data = []
                for batch_data, batch_targets in test_data:
                    new_data.append(([tb[:, :, :r] for tb in batch_data],
                                     [tb[:, :, 0] for tb in batch_targets]))
                test_data = new_data
            else:
                print("Using only harmony")
                new_data = []
                for batch_data, batch_targets in test_data:
                    new_data.append(([tb[:, :, r:] for tb in batch_data],
                                     [tb[:, :, 1] for tb in batch_targets]))
                test_data = new_data
        else:
            model_class = NottinghamModel
            # use time batch len of 1 so that every target is covered
            test_data = util.batch_data(data_pickle['test'], time_batch_len = 1, 
                max_time_batches = -1, softmax = True)
    else:
        raise Exception("Other datasets not yet implemented")
        
    print(config)

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            test_model = model_class(config, training=False)

        saver = tf.train.Saver(tf.global_variables())
        model_path = os.path.join(os.path.dirname(args.config_file), 
            config.model_name)
        if not tf.train.checkpoint_exists(model_path):
            raise FileNotFoundError("Checkpoint not found for model base path: {}".format(model_path))
        saver.restore(session, model_path)
        
        test_loss, test_probs = util.run_epoch(session, test_model, test_data, 
            training=False, testing=True)
        print('Testing Loss: {}'.format(test_loss))

        if config.dataset == 'softmax':
            if separate_mode:
                nottingham_util.seperate_accuracy(test_probs, test_data, num_samples=args.num_samples)
            else:
                nottingham_util.accuracy(test_probs, test_data, num_samples=args.num_samples)

        else:
            util.accuracy(test_probs, test_data, num_samples=50)

    sys.exit(0)
