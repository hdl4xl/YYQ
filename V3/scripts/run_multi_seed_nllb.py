from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_SEEDS = [42, 13, 2026]
MODEL_NAME = "facebook/nllb-200-distilled-600M"


def seed_label(seed: int) -> str:
    return f"{seed:04d}"


def run_dir_for(root: Path, seed: int) -> Path:
    return root / "runs" / f"nllb_seed_{seed_label(seed)}"


def has_complete_run(run_dir: Path) -> bool:
    required = [
        run_dir / "metrics" / "metrics_summary.csv",
        run_dir / "metrics" / "ablation_summary.csv",
        run_dir / "metrics" / "entity_type_metrics.csv",
        run_dir / "metrics" / "evaluation_status.json",
        run_dir / "paper_tables" / "results.md",
    ]
    translations = [
        run_dir / "translations" / "B1_general_translation.csv",
        run_dir / "translations" / "B2_domain_adapted.csv",
        run_dir / "translations" / "B3_entity_aware.csv",
        run_dir / "translations" / "Ours_entity_aware_domain_adapted.csv",
    ]
    if not all(path.exists() for path in required + translations):
        return False
    try:
        status = json.loads((run_dir / "metrics" / "evaluation_status.json").read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return bool(status.get("success"))


def build_command(args: argparse.Namespace, root: Path, seed: int) -> list[str]:
    return [
        args.conda,
        "run",
        "-n",
        args.env_name,
        "python",
        str(root / "scripts" / "nllb_tcm_experiment.py"),
        "all",
        "--root",
        str(root / "YYQEI_20260508_NLLB_ready"),
        "--run-dir",
        str(run_dir_for(root, seed)),
        "--model-name",
        MODEL_NAME,
        "--download-retries",
        str(args.download_retries),
        "--epochs",
        str(args.epochs),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--infer-batch-size",
        str(args.infer_batch_size),
        "--seed",
        str(seed),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V3 NLLB experiments for multiple training seeds.")
    parser.add_argument("--conda", default=r"D:\Anaconda\Scripts\conda.exe")
    parser.add_argument("--env-name", default="YYQEI")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--force", action="store_true", help="Re-run even if a seed already has complete outputs.")
    parser.add_argument("--download-retries", type=int, default=20)
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--infer-batch-size", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    failures: list[int] = []
    for seed in args.seeds:
        run_dir = run_dir_for(root, seed)
        if has_complete_run(run_dir) and not args.force:
            print(f"[multi-seed] seed={seed} already complete: {run_dir}")
            continue
        cmd = build_command(args, root, seed)
        if args.force:
            cmd.append("--force")
        print(f"[multi-seed] running seed={seed} -> {run_dir}")
        code = subprocess.run(cmd, cwd=root).returncode
        if code:
            print(f"[multi-seed] FAILED seed={seed} exit_code={code}")
            failures.append(seed)
        elif not has_complete_run(run_dir):
            print(f"[multi-seed] FAILED seed={seed}: required outputs incomplete")
            failures.append(seed)
        else:
            print(f"[multi-seed] complete seed={seed}")

    if failures:
        raise SystemExit(f"Failed seeds: {failures}")
    print("[multi-seed] all requested seeds complete")


if __name__ == "__main__":
    main()
