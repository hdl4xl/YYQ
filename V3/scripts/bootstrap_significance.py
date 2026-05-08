from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from sacrebleu.metrics import BLEU, CHRF, TER

from nllb_tcm_experiment import contains_term, parse_entities, read_csv, unique_entities, write_csv


DEFAULT_SEEDS = [42, 13, 2026]
MODEL_IDS = {
    "B2": "B2_domain_adapted",
    "B3": "B3_entity_aware",
    "Ours": "Ours_entity_aware_domain_adapted",
}
METRICS = ["BLEU", "chrF", "TER", "TA", "TCR"]
SACREBLEU_METRICS = {
    "BLEU": BLEU(effective_order=True),
    "chrF": CHRF(),
    "TER": TER(),
}


def seed_label(seed: int) -> str:
    return f"{seed:04d}"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return ordered[int(k)]
    return ordered[lower] * (upper - k) + ordered[upper] * (k - lower)


def read_translation(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def align_rows(ours: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_id = {row["id"]: row for row in baseline}
    aligned_ours: list[dict[str, Any]] = []
    aligned_base: list[dict[str, Any]] = []
    for row in ours:
        row_id = row["id"]
        if row_id not in baseline_by_id:
            raise ValueError(f"Missing baseline row id={row_id}")
        base = baseline_by_id[row_id]
        if row.get("reference") != base.get("reference"):
            raise ValueError(f"Reference mismatch for id={row_id}")
        aligned_ours.append(row)
        aligned_base.append(base)
    return aligned_ours, aligned_base


def precompute_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    refs = [row["reference"] for row in rows]
    preds = [row["prediction"] for row in rows]
    metric_stats: dict[str, list[list[float]]] = {}
    for metric_name, metric in SACREBLEU_METRICS.items():
        metric_stats[metric_name] = metric._extract_corpus_statistics(preds, [refs])

    term_rows: list[list[tuple[str, int]]] = []
    for row in rows:
        pred = row["prediction"]
        entities = unique_entities(parse_entities(row.get("entities_eval")) or parse_entities(row.get("entities")))
        row_terms: list[tuple[str, int]] = []
        for ent in entities:
            row_terms.append((ent["zh"], int(contains_term(pred, ent["en"]))))
        term_rows.append(row_terms)
    return {"metric_stats": metric_stats, "term_rows": term_rows, "n": len(rows)}


def score_precomputed(precomp: dict[str, Any], indexes: list[int]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for metric_name, metric in SACREBLEU_METRICS.items():
        stats_rows = precomp["metric_stats"][metric_name]
        width = len(stats_rows[0])
        totals = [0.0] * width
        for idx in indexes:
            stats = stats_rows[idx]
            for pos, value in enumerate(stats):
                totals[pos] += value
        scores[metric_name] = metric._compute_score_from_stats(totals).score

    total = 0
    hit = 0
    term_occurrences: dict[str, list[int]] = defaultdict(list)
    term_rows = precomp["term_rows"]
    for idx in indexes:
        for zh, matched in term_rows[idx]:
            total += 1
            hit += matched
            term_occurrences[zh].append(matched)
    repeated_hits = 0
    repeated_total = 0
    for values in term_occurrences.values():
        if len(values) >= 2:
            repeated_hits += sum(values)
            repeated_total += len(values)
    ta = hit / total * 100 if total else 0.0
    tcr = repeated_hits / repeated_total * 100 if repeated_total else 0.0
    return {
        "BLEU": scores["BLEU"],
        "chrF": scores["chrF"],
        "TER": scores["TER"],
        "TA": ta,
        "TCR": tcr,
    }


def metric_delta(ours_score: dict[str, float], base_score: dict[str, float], metric: str) -> float:
    if metric == "TER":
        return base_score[metric] - ours_score[metric]
    return ours_score[metric] - base_score[metric]


def run_comparison(
    seed: int,
    comparison_name: str,
    ours_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    iterations: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    ours_rows, base_rows = align_rows(ours_rows, base_rows)
    n = len(ours_rows)
    all_indexes = list(range(n))
    ours_precomp = precompute_scores(ours_rows)
    base_precomp = precompute_scores(base_rows)
    observed_ours = score_precomputed(ours_precomp, all_indexes)
    observed_base = score_precomputed(base_precomp, all_indexes)
    boot_deltas = {metric: [] for metric in METRICS}

    for _ in range(iterations):
        indexes = [rng.randrange(n) for _ in range(n)]
        sample_ours_score = score_precomputed(ours_precomp, indexes)
        sample_base_score = score_precomputed(base_precomp, indexes)
        for metric in METRICS:
            boot_deltas[metric].append(metric_delta(sample_ours_score, sample_base_score, metric))

    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        deltas = boot_deltas[metric]
        observed_delta = metric_delta(observed_ours, observed_base, metric)
        p_value = (sum(1 for value in deltas if value <= 0.0) + 1) / (len(deltas) + 1)
        rows.append(
            {
                "seed": seed,
                "comparison": comparison_name,
                "metric": metric,
                "n": n,
                "iterations": iterations,
                "baseline_score": round(observed_base[metric], 4),
                "ours_score": round(observed_ours[metric], 4),
                "observed_delta": round(observed_delta, 4),
                "bootstrap_mean_delta": round(mean(deltas), 4),
                "ci_low": round(percentile(deltas, 2.5), 4),
                "ci_high": round(percentile(deltas, 97.5), 4),
                "p_value": round(p_value, 6),
                "significant_p05": str(p_value < 0.05),
                "delta_direction": "baseline_minus_ours" if metric == "TER" else "ours_minus_baseline",
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = ["seed", "comparison", "metric", "observed_delta", "ci_low", "ci_high", "p_value", "significant_p05"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap significance for V3 NLLB seed runs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260508)
    parser.add_argument("--output-dir", default="analysis/bootstrap_significance")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_dir = root / "runs" / f"nllb_seed_{seed_label(seed)}"
        trans_dir = run_dir / "translations"
        if not trans_dir.exists():
            raise FileNotFoundError(f"Missing translations directory for seed={seed}: {trans_dir}")
        ours_rows = read_translation(trans_dir / f"{MODEL_IDS['Ours']}.csv")
        base_rows = {
            "Ours_vs_B2": read_translation(trans_dir / f"{MODEL_IDS['B2']}.csv"),
            "Ours_vs_B3": read_translation(trans_dir / f"{MODEL_IDS['B3']}.csv"),
        }
        seed_rng = random.Random(args.bootstrap_seed + seed)
        seed_rows: list[dict[str, Any]] = []
        for comparison_name, baseline in base_rows.items():
            rows = run_comparison(seed, comparison_name, ours_rows, baseline, args.iterations, seed_rng)
            seed_rows.extend(rows)
            all_rows.extend(rows)
        seed_path = output_dir / f"bootstrap_seed_{seed_label(seed)}.csv"
        write_csv(seed_path, seed_rows, list(seed_rows[0].keys()))
        print(f"[bootstrap] wrote {seed_path}")

    summary_path = output_dir / "bootstrap_summary.csv"
    write_csv(summary_path, all_rows, list(all_rows[0].keys()))
    md = [
        "# V3 Paired Bootstrap Significance",
        "",
        f"- Iterations: {args.iterations}",
        f"- Bootstrap random seed: {args.bootstrap_seed}",
        "- Positive delta means Ours is better. For TER, delta is baseline TER minus Ours TER.",
        "",
        markdown_table(all_rows),
        "",
    ]
    md_path = output_dir / "bootstrap_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8-sig")
    meta_path = output_dir / "bootstrap_config.json"
    meta_path.write_text(
        json.dumps({"seeds": args.seeds, "iterations": args.iterations, "bootstrap_seed": args.bootstrap_seed}, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    print(f"[bootstrap] wrote {summary_path}")
    print(f"[bootstrap] wrote {md_path}")


if __name__ == "__main__":
    main()
