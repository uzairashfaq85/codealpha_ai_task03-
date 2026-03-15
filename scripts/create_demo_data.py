"""
Project: codealpha_ai_task03-
Created: August 2024
Description: Create a small synthetic Nottingham-style dataset for quick local validation.
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from music_rnn import nottingham_util


def make_sequence(length, chord_to_idx):
    melody_range = nottingham_util.NOTTINGHAM_MELODY_RANGE
    num_chords = len(chord_to_idx)
    dim = melody_range + num_chords

    sequence = np.zeros((length, dim), dtype=np.float32)
    chord_names = ["CM", "Am", "FM", "GM", nottingham_util.NO_CHORD]

    for step in range(length):
        melody_idx = random.randint(0, melody_range - 2)
        chord_name = chord_names[(step // 8) % len(chord_names)]
        chord_idx = chord_to_idx[chord_name]

        sequence[step, melody_idx] = 1.0
        sequence[step, melody_range + chord_idx] = 1.0

    return sequence


def main():
    parser = argparse.ArgumentParser(description="Create synthetic nottingham.pickle for quick testing")
    parser.add_argument("--output", type=str, default="data/nottingham.pickle")
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--valid", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--length", type=int, default=256)
    args = parser.parse_args()

    random.seed(7)
    np.random.seed(7)

    chord_to_idx = {
        "CM": 0,
        "Am": 1,
        "FM": 2,
        "GM": 3,
        nottingham_util.NO_CHORD: 4,
    }

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def build_split(count, split_name):
        seqs = [make_sequence(args.length, chord_to_idx) for _ in range(count)]
        metadata = [{"name": f"{split_name}_{i}", "path": f"synthetic/{split_name}_{i}.mid"} for i in range(count)]
        return seqs, metadata

    train, train_meta = build_split(args.train, "train")
    valid, valid_meta = build_split(args.valid, "valid")
    test, test_meta = build_split(args.test, "test")

    store = {
        "chord_to_idx": chord_to_idx,
        "train": train,
        "valid": valid,
        "test": test,
        "train_metadata": train_meta,
        "valid_metadata": valid_meta,
        "test_metadata": test_meta,
    }

    with open(output_path, "wb") as handle:
        pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Synthetic dataset created at: {output_path}")
    print(f"Train/Valid/Test sizes: {len(train)}/{len(valid)}/{len(test)}")


if __name__ == "__main__":
    main()
