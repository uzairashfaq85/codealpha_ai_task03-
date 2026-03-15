"""
Project: codealpha_ai_task03-
Created: August 2024
Description: Main training entrypoint for the Nottingham dual-softmax model.
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
from music_rnn.model import NottinghamModel

tf.disable_v2_behavior()

def get_config_name(config):
    def replace_dot(s): return s.replace(".", "p")
    return "nl_" + str(config.num_layers) + "_hs_" + str(config.hidden_size) + \
            replace_dot("_mc_{}".format(config.melody_coeff)) + \
            replace_dot("_dp_{}".format(config.dropout_prob)) + \
            replace_dot("_idp_{}".format(config.input_dropout_prob)) + \
            replace_dot("_tb_{}".format(config.time_batch_len)) 

class DefaultConfig(object):
    # model parameters
    num_layers = 2
    hidden_size = 200
    melody_coeff = 0.5
    dropout_prob = 0.5
    input_dropout_prob = 0.8
    cell_type = 'lstm'

    # learning parameters
    max_time_batches = 9 
    time_batch_len = 128
    learning_rate = 5e-3
    learning_rate_decay = 0.9
    num_epochs = 250

    # metadata
    dataset = 'softmax'
    model_file = ''

    def __repr__(self):
        return """Num Layers: {}, Hidden Size: {}, Melody Coeff: {}, Dropout Prob: {}, Input Dropout Prob: {}, Cell Type: {}, Time Batch Len: {}, Learning Rate: {}, Decay: {}""".format(self.num_layers, self.hidden_size, self.melody_coeff, self.dropout_prob, self.input_dropout_prob, self.cell_type, self.time_batch_len, self.learning_rate, self.learning_rate_decay)
    
def train_model():
    np.random.seed()

    parser = argparse.ArgumentParser(description='Script to train and save a model.')
    parser.add_argument('--dataset', type=str, default='softmax',
                        # choices = ['bach', 'nottingham', 'softmax'],
                        choices = ['softmax'])
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
        model_class = NottinghamModel
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

    grid = {
        "dropout_prob": [0.5],
        "input_dropout_prob": [0.8],
        "melody_coeff": [0.5],
        "num_layers": [2],
        "hidden_size": [200],
        "num_epochs": [250],
        "learning_rate": [5e-3],
        "learning_rate_decay": [0.9],
        "time_batch_len": [128],
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

        logger.info(config)
        config_file_path = os.path.join(run_folder, get_config_name(config) + '.config')
        with open(config_file_path, 'wb') as f:
            pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)

        with tf.Graph().as_default(), tf.Session() as session:
            with tf.variable_scope("model", reuse=None):
                train_graph_model = model_class(config, training=True)
            with tf.variable_scope("model", reuse=True):
                valid_model = model_class(config, training=False)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)
            tf.global_variables_initializer().run()

            # training
            early_stop_best_loss = None
            start_saving = False
            saved_flag = False
            train_losses, valid_losses = [], []
            start_time = time.time()
            for i in range(config.num_epochs):
                loss = util.run_epoch(session, train_graph_model, 
                    data["train"]["data"], training=True, testing=False)
                train_losses.append((i, loss))
                if i == 0:
                    continue

                logger.info('Epoch: {}, Train Loss: {}, Time Per Epoch: {}'.format(\
                        i, loss, (time.time() - start_time)/i))
                valid_loss = util.run_epoch(session, valid_model, data["valid"]["data"], training=False, testing=False)
                valid_losses.append((i, valid_loss))
                logger.info('Valid Loss: {}'.format(valid_loss))

                if early_stop_best_loss == None: #prevents overfitting 
                    early_stop_best_loss = valid_loss
                elif valid_loss < early_stop_best_loss:
                    early_stop_best_loss = valid_loss
                    if start_saving:
                        logger.info('Best loss so far encountered, saving model.')
                        saver.save(session, os.path.join(run_folder, config.model_name))
                        saved_flag = True
                elif not start_saving:
                    start_saving = True 
                    logger.info('Valid loss increased for the first time, will start saving models') #training decreases always but valid loss increases after
                    saver.save(session, os.path.join(run_folder, config.model_name))                 #certain point so start saving models to prevent overfitting
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


if __name__ == '__main__':
    train_model()
