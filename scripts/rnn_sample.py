"""
Project: codealpha_ai_task03-
Created: August 2024
Description: Generate a MIDI sample from a trained model checkpoint.
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

from music_rnn import nottingham_util
from music_rnn.model import NottinghamModel
from rnn import DefaultConfig

tf.disable_v2_behavior()


def load_config(config_file):
    import __main__

    __main__.DefaultConfig = DefaultConfig
    with open(config_file, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    np.random.seed()      

    parser = argparse.ArgumentParser(description='Script to generated a MIDI file sample from a trained model.')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--pickle_path', type=str, default=nottingham_util.PICKLE_LOC)
    parser.add_argument('--output_midi', type=str, default='best.midi')
    parser.add_argument('--sample_melody', action='store_true', default=False)
    parser.add_argument('--sample_harmony', action='store_true', default=False)
    parser.add_argument('--sample_seq', type=str, default='random',
        choices = ['random', 'chords'])
    parser.add_argument('--conditioning', type=int, default=-1)
    parser.add_argument('--sample_length', type=int, default=512)

    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        raise FileNotFoundError("Config file not found: {}".format(args.config_file))

    config = load_config(args.config_file)

    if config.dataset == 'softmax':
        config.time_batch_len = 1
        config.max_time_batches = -1
        model_class = NottinghamModel
        if not os.path.exists(args.pickle_path):
            raise FileNotFoundError(
                "Dataset pickle not found: {}. Run scripts/create_demo_data.py or scripts/main.py first.".format(
                    args.pickle_path
                )
            )
        with open(args.pickle_path, 'rb') as f:
            data_pickle = pickle.load(f)
        chord_to_idx = data_pickle['chord_to_idx']

        time_step = 120
        resolution = 480

    else:
        raise Exception("Other datasets not yet implemented")

    print(config)

    sample_seq = []

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            sampling_model = model_class(config)

        saver = tf.train.Saver(tf.global_variables())
        model_path = os.path.join(os.path.dirname(args.config_file), 
            config.model_name)
        if not tf.train.checkpoint_exists(model_path):
            raise FileNotFoundError("Checkpoint not found for model base path: {}".format(model_path))
        saver.restore(session, model_path)

        state = sampling_model.get_cell_zero_state(session, 1)
        if args.sample_seq == 'chords':
            # 16 - one measure, 64 - chord progression
            repeats = args.sample_length // 64
            sample_seq = nottingham_util.i_vi_iv_v(chord_to_idx, repeats, config.input_dim)
            print('Sampling melody using a I, VI, IV, V progression')

        elif args.sample_seq == 'random':
            sample_index = np.random.choice(np.arange(len(data_pickle['test'])))
            sample_seq = [
                data_pickle['test'][sample_index][i, :]
                for i in range(data_pickle['test'][sample_index].shape[0])
            ]

        if not sample_seq:
            raise ValueError("No sample sequence available for generation")

        chord = sample_seq[0]
        seq = [chord]

        if args.conditioning > 0:
            for i in range(1, args.conditioning):
                seq_input = np.reshape(chord, [1, 1, config.input_dim])
                feed = {
                    sampling_model.seq_input: seq_input,
                    sampling_model.initial_state: state,
                }
                state = session.run(sampling_model.final_state, feed_dict=feed)
                chord = sample_seq[i]
                seq.append(chord)

        if config.dataset == 'softmax':
            writer = nottingham_util.NottinghamMidiWriter(chord_to_idx, verbose=False)
            sampler = nottingham_util.NottinghamSampler(chord_to_idx, verbose=False)
        else:
            # writer = midi_util.MidiWriter()
            # sampler = sampling.Sampler(verbose=False)
            raise Exception("Other datasets not yet implemented")

        for i in range(max(args.sample_length - len(seq), 0)):
            seq_input = np.reshape(chord, [1, 1, config.input_dim])
            feed = {
                sampling_model.seq_input: seq_input,
                sampling_model.initial_state: state,
            }
            [probs, state] = session.run(
                [sampling_model.probs, sampling_model.final_state],
                feed_dict=feed)
            probs = np.reshape(probs, [config.input_dim])
            chord = sampler.sample_notes(probs)

            if config.dataset == 'softmax':
                r = nottingham_util.NOTTINGHAM_MELODY_RANGE
                if args.sample_melody:
                    chord[r:] = 0
                    chord[r:] = sample_seq[i][r:]
                elif args.sample_harmony:
                    chord[:r] = 0
                    chord[:r] = sample_seq[i][:r]

            seq.append(chord)

        writer.dump_sequence_to_midi(seq, args.output_midi, 
            time_step=time_step, resolution=resolution)
