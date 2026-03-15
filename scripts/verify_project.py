"""
Project: codealpha_ai_task03-
Created: August 2024
Description: Lightweight project verification script for syntax/import/CLI readiness.
"""

import importlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
SRC_DIR = os.path.join(ROOT, "src")


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def main():
    checks = []

    code, out, err = run_cmd([sys.executable, "-m", "compileall", "scripts", "src"])
    checks.append(("Compile all Python files", code == 0, out or err))

    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    for module_name in [
        "music_rnn",
        "music_rnn.model",
        "music_rnn.util",
        "music_rnn.midi_util",
        "music_rnn.nottingham_util",
        "music_rnn.sampling",
    ]:
        try:
            importlib.import_module(module_name)
            checks.append((f"Import {module_name}", True, "OK"))
        except Exception as exc:
            checks.append((f"Import {module_name}", False, str(exc)))

    for script in ["main.py", "rnn.py", "rnn_separate.py", "rnn_test.py", "rnn_sample.py", "create_demo_data.py"]:
        code, out, err = run_cmd([sys.executable, os.path.join(SCRIPTS_DIR, script), "--help"])
        checks.append((f"CLI help {script}", code == 0, out or err))

    print("=" * 72)
    print("Project Verification Report")
    print("=" * 72)

    failed = 0
    for name, ok, details in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if not ok:
            failed += 1
            print(f"       {details.splitlines()[:1][0] if details else 'No details'}")

    print("=" * 72)
    if failed == 0:
        print("All checks passed.")
        return 0

    print(f"{failed} check(s) failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
