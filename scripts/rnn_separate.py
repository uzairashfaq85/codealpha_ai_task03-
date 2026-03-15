"""
Project: codealpha_ai_task03-
Created: August 2024
Description: Train baseline melody-only or harmony-only softmax model variants.
"""

import argparse
import itertools
import logging
import os
import pickle
import random
import string
import sys
import time
 
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from music_rnn import nottingham_util
from music_rnn import util
from rnn import get_config_name, DefaultConfig
from music_rnn.model import NottinghamSeparate

tf.disable_v2_behavior()

if __name__ == '__main__':
    np.random.seed()      

    parser = argparse.ArgumentParser(description='Music RNN')
    parser.add_argument('--choice', type=str, default='melody',
                        choices = ['melody', 'harmony'])
    parser.add_argument('--dataset', type=str, default='softmax',
                        choices = ['bach', 'nottingham', 'softmax'])
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--run_name', type=str, default=time.strftime("%m%d_%H%M"))
    parser.add_argument('--pickle_path', type=str, default=nottingham_util.PICKLE_LOC)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--max_time_batches', type=int, default=None)
    parser.add_argument('--time_batch_len', type=int, default=None)
    parser.add_argument('--fast_dev_run', action='store_true', default=False)

    args = parser.parse_args()

    if args.dataset == 'softmax':
        time_step = 120
        model_class = NottinghamSeparate
        if not os.path.exists(args.pickle_path):
            raise FileNotFoundError(
                "Dataset pickle not found: {}. Run scripts/create_demo_data.py or scripts/main.py first.".format(
                    args.pickle_path
                )
            )
        with open(args.pickle_path, 'rb') as f:
            data_pickle = pickle.load(f)

        input_dim = data_pickle["train"][0].shape[1]
        print('Finished loading data, input dim: {}'.format(input_dim))
    else:
        raise Exception("Other datasets not yet implemented")

    best_config = None
    best_valid_loss = None

    # set up run dir
    run_folder = os.path.join(args.model_dir, args.run_name)
    if os.path.exists(run_folder):
        raise Exception("Run name {} already exists, choose a different one".format(run_folder))
    os.makedirs(run_folder)

    logger = logging.getLogger(__name__) 
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join(run_folder, "training.log")))

    # grid
    grid = {
        "dropout_prob": [0.65],
        "input_dropout_prob": [0.9],
        "num_layers": [1],
        "hidden_size": [100]
    }

    # Generate product of hyperparams
    grid_keys = list(grid.keys())
    runs = [list(zip(grid_keys, x)) for x in itertools.product(*grid.values())]
    logger.info("{} runs detected".format(len(runs)))

    for combination in runs:

        config = DefaultConfig()
        config.dataset = args.dataset
        config.model_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(12)) + '.model'
        for attr, value in combination:
            setattr(config, attr, value)

        if args.fast_dev_run:
            config.num_epochs = 2
            config.max_time_batches = -1
            config.time_batch_len = 32
            config.hidden_size = 64
        if args.num_epochs is not None:
            config.num_epochs = args.num_epochs
        if args.max_time_batches is not None:
            config.max_time_batches = args.max_time_batches
        if args.time_batch_len is not None:
            config.time_batch_len = args.time_batch_len

        if config.dataset == 'softmax':
            data = util.load_data('', time_step, config.time_batch_len, config.max_time_batches, nottingham=data_pickle)
            config.input_dim = data["input_dim"]
        else:
            raise Exception("Other datasets not yet implemented")

        # cut away unnecessary parts
        r = nottingham_util.NOTTINGHAM_MELODY_RANGE
        if args.choice == 'melody':
            print("Using only melody")
            for d in ['train', 'test', 'valid']:
                new_data = []
                for batch_data, batch_targets in data[d]["data"]:
                    new_data.append(([tb[:, :, :r] for tb in batch_data],
                                     [tb[:, :, 0] for tb in batch_targets]))
                data[d]["data"] = new_data
        else:
            print("Using only harmony")
            for d in ['train', 'test', 'valid']:
                new_data = []
                for batch_data, batch_targets in data[d]["data"]:
                    new_data.append(([tb[:, :, r:] for tb in batch_data],
                                     [tb[:, :, 1] for tb in batch_targets]))
                data[d]["data"] = new_data

        input_dim = data["input_dim"] = data["train"]["data"][0][0][0].shape[2]
        config.input_dim = input_dim
        print("New input dim: {}".format(input_dim))

        logger.info(config)
        config_file_path = os.path.join(run_folder, get_config_name(config) + '.config')
        with open(config_file_path, 'wb') as f:
            pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)

        with tf.Graph().as_default(), tf.Session() as session:
            with tf.variable_scope("model", reuse=None):
                train_model = model_class(config, training=True)
            with tf.variable_scope("model", reuse=True):
                valid_model = model_class(config, training=False)

            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()

            # training
            early_stop_best_loss = None
            start_saving = False
            saved_flag = False
            train_losses, valid_losses = [], []
            start_time = time.time()
            for i in range(config.num_epochs):
                loss = util.run_epoch(session, train_model, data["train"]["data"], training=True, testing=False)
                train_losses.append((i, loss))
                if i == 0:
                    continue

                valid_loss = util.run_epoch(session, valid_model, data["valid"]["data"], training=False, testing=False)
                valid_losses.append((i, valid_loss))

                logger.info('Epoch: {}, Train Loss: {}, Valid Loss: {}, Time Per Epoch: {}'.format(\
                        i, loss, valid_loss, (time.time() - start_time)/i))

                # if it's best validation loss so far, save it
                if early_stop_best_loss == None:
                    early_stop_best_loss = valid_loss
                elif valid_loss < early_stop_best_loss:
                    early_stop_best_loss = valid_loss
                    if start_saving:
                        logger.info('Best loss so far encountered, saving model.')
                        saver.save(session, os.path.join(run_folder, config.model_name))
                        saved_flag = True
                elif not start_saving:
                    start_saving = True 
                    logger.info('Valid loss increased for the first time, will start saving models')
                    saver.save(session, os.path.join(run_folder, config.model_name))
                    saved_flag = True

            if not saved_flag:
                saver.save(session, os.path.join(run_folder, config.model_name))

            # set loss axis max to 20
            axes = plt.gca()
            if config.dataset == 'softmax':
                axes.set_ylim([0, 2])
            else:
                axes.set_ylim([0, 100])
            plt.plot([t[0] for t in train_losses], [t[1] for t in train_losses])
            plt.plot([t[0] for t in valid_losses], [t[1] for t in valid_losses])
            plt.legend(['Train Loss', 'Validation Loss'])
            chart_file_path = os.path.join(run_folder, get_config_name(config) + '.png')
            plt.savefig(chart_file_path)
            plt.clf()

            logger.info("Config {}, Loss: {}".format(config, early_stop_best_loss))
            if best_valid_loss == None or early_stop_best_loss < best_valid_loss:
                logger.info("Found best new model!")
                best_valid_loss = early_stop_best_loss
                best_config = config

    logger.info("Best Config: {}, Loss: {}".format(best_config, best_valid_loss))
