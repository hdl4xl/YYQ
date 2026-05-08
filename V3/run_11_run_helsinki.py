from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict no-leak Helsinki experiment.")
    parser.add_argument("--conda", default=r"D:\Anaconda\Scripts\conda.exe")
    parser.add_argument("--env-name", default="YYQEI")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ready_root = root / "YYQEI_20260508_NLLB_ready"
    run_dir = root / "runs" / "helsinki_20260508_strict"
    cmd = [
        args.conda,
        "run",
        "-n",
        args.env_name,
        "python",
        str(root / "scripts" / "nllb_tcm_experiment.py"),
        "all",
        "--root",
        str(ready_root),
        "--run-dir",
        str(run_dir),
        "--model-name",
        "Helsinki-NLP/opus-mt-zh-en",
        "--download-retries",
        "20",
        "--epochs",
        "6",
        "--train-batch-size",
        "2",
        "--eval-batch-size",
        "2",
        "--gradient-accumulation-steps",
        "8",
        "--infer-batch-size",
        "4",
    ]
    raise SystemExit(subprocess.run(cmd, cwd=root).returncode)


if __name__ == "__main__":
    main()
