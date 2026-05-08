from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def conda_python(conda: str, env_name: str, args: list[str], cwd: Path) -> int:
    cmd = [conda, "run", "-n", env_name, "python", *args]
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Helsinki and NLLB experiment environments.")
    parser.add_argument("--conda", default=r"D:\Anaconda\Scripts\conda.exe")
    parser.add_argument("--env-name", default="YYQEI")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ready_root = root / "YYQEI_20260508_NLLB_ready"
    checks = [
        ("facebook/nllb-200-distilled-600M", root / "runs" / "check_nllb"),
        ("Helsinki-NLP/opus-mt-zh-en", root / "runs" / "check_helsinki"),
    ]
    for model_name, run_dir in checks:
        code = conda_python(
            args.conda,
            args.env_name,
            [
                str(root / "scripts" / "nllb_tcm_experiment.py"),
                "check-env",
                "--root",
                str(ready_root),
                "--run-dir",
                str(run_dir),
                "--model-name",
                model_name,
            ],
            root,
        )
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
