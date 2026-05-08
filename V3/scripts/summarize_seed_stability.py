from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from nllb_tcm_experiment import read_csv, write_csv


DEFAULT_SEEDS = [42, 13, 2026]
METRICS = ["BLEU", "chrF", "TER", "TA", "TCR"]


def seed_label(seed: int) -> str:
    return f"{seed:04d}"


def load_seed_metrics(root: Path, seed: int) -> list[dict[str, Any]]:
    path = root / "runs" / f"nllb_seed_{seed_label(seed)}" / "metrics" / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics for seed={seed}: {path}")
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No metrics rows for seed={seed}: {path}")
    for row in rows:
        row["seed"] = seed
    return rows


def format_mean_std(mu: float, sigma: float) -> str:
    return f"{mu:.4f} ± {sigma:.4f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = ["model_id", "n_seeds", "BLEU", "chrF", "TER", "TA", "TCR"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V3 multi-seed metric stability.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", default="analysis/seed_stability")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        long_rows.extend(load_seed_metrics(root, seed))

    long_fields = ["seed", "model_id", "n", "BLEU", "chrF", "TER", "TA", "TCR", "normalization_used"]
    write_csv(output_dir / "seed_metrics_long.csv", long_rows, long_fields)

    model_ids = sorted({row["model_id"] for row in long_rows})
    summary_rows: list[dict[str, Any]] = []
    md_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        rows = [row for row in long_rows if row["model_id"] == model_id]
        summary: dict[str, Any] = {"model_id": model_id, "n_seeds": len(rows)}
        md_row: dict[str, Any] = {"model_id": model_id, "n_seeds": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            sigma = stdev(values) if len(values) >= 2 else 0.0
            summary[f"{metric}_mean"] = round(mean(values), 4)
            summary[f"{metric}_std"] = round(sigma, 4)
            summary[f"{metric}_mean_std"] = format_mean_std(mean(values), sigma)
            md_row[metric] = summary[f"{metric}_mean_std"]
        summary_rows.append(summary)
        md_rows.append(md_row)

    summary_fields = ["model_id", "n_seeds"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"])
    write_csv(output_dir / "seed_stability_summary.csv", summary_rows, summary_fields)

    md = [
        "# V3 Multi-Seed Stability",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in args.seeds)}",
        "- Values are mean ± sample standard deviation across seeds.",
        "",
        markdown_table(md_rows),
        "",
    ]
    md_path = output_dir / "seed_stability_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8-sig")
    print(f"[seed-stability] wrote {output_dir / 'seed_metrics_long.csv'}")
    print(f"[seed-stability] wrote {output_dir / 'seed_stability_summary.csv'}")
    print(f"[seed-stability] wrote {md_path}")


if __name__ == "__main__":
    main()
