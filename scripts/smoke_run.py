"""
Project: codealpha_ai_task03-
Created: August 2024
Description: End-to-end smoke test: demo data -> train -> test -> sample.
"""

import argparse
import glob
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")


def run_step(name, cmd):
    print("=" * 80)
    print(name)
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError("Step failed: {}".format(name))


def main():
    parser = argparse.ArgumentParser(description="Run full smoke test for the project")
    parser.add_argument("--run_name", type=str, default="smoke_" + time.strftime("%H%M%S"))
    parser.add_argument("--sample_length", type=int, default=64)
    args = parser.parse_args()

    run_step(
        "Create synthetic dataset",
        [sys.executable, os.path.join(SCRIPTS_DIR, "create_demo_data.py")],
    )

    run_step(
        "Train model (fast dev run)",
        [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "rnn.py"),
            "--run_name",
            args.run_name,
            "--model_dir",
            "models",
            "--fast_dev_run",
        ],
    )

    config_pattern = os.path.join(ROOT, "models", args.run_name, "*.config")
    configs = sorted(glob.glob(config_pattern))
    if not configs:
        raise FileNotFoundError("No config file produced under {}".format(config_pattern))
    config_file = configs[0]

    run_step(
        "Evaluate trained model",
        [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "rnn_test.py"),
            "--config_file",
            config_file,
            "--num_samples",
            "1",
        ],
    )

    run_step(
        "Generate sample MIDI",
        [
            sys.executable,
            os.path.join(SCRIPTS_DIR, "rnn_sample.py"),
            "--config_file",
            config_file,
            "--sample_length",
            str(args.sample_length),
            "--output_midi",
            "smoke_sample.midi",
        ],
    )

    print("=" * 80)
    print("Smoke run completed successfully.")
    print("Run folder: models/{}".format(args.run_name))
    print("Sample MIDI: smoke_sample.midi")


if __name__ == "__main__":
    main()
