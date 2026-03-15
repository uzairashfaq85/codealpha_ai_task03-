"""
Project: codealpha_ai_task03-
Created: August 2024
Description: Dataset bootstrap script to download Nottingham data and prepare training pickle.
"""

import argparse
import os
import sys
import urllib.request
import zipfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from music_rnn import nottingham_util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("NumTracks", type=int)
    parser.add_argument("--dataset_url", type=str, default="http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip")
    parser.add_argument("--dataset_zip", type=str, default="dataset.zip")
    parser.add_argument("--skip_download", action="store_true", default=False)
    args = parser.parse_args()

    if not args.skip_download:
        urllib.request.urlretrieve(args.dataset_url, args.dataset_zip)
    elif not os.path.exists(args.dataset_zip):
        raise FileNotFoundError(
            "--skip_download was set but dataset zip file was not found: {}".format(args.dataset_zip)
        )

    with zipfile.ZipFile(args.dataset_zip) as zip_file:
        zip_file.extractall("data")

    nottingham_util.create_model(args.NumTracks)


if __name__ == "__main__":
    main()
