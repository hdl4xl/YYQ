from __future__ import annotations

import csv
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


SPLITS = ("train", "dev", "test")
FULLWIDTH_SEMICOLON = "\uff1b"
STRICT_KEEP_PRIORITY = {"train": 0, "dev": 1, "test": 2}


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm_zh(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff" or ch.isascii() and ch.isalnum())


def norm_en(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def contains_en(reference: str, term_en: str) -> bool:
    return str(term_en).lower() in str(reference).lower()


def parse_entity_string(value: Any) -> list[dict[str, str]]:
    if pd.isna(value) or not str(value).strip():
        return []
    entities: list[dict[str, str]] = []
    for part in str(value).replace(FULLWIDTH_SEMICOLON, ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if " / " in part:
            zh, en = part.split(" / ", 1)
        elif "/" in part:
            zh, en = part.split("/", 1)
        else:
            zh, en = part, ""
        entities.append({"zh": zh.strip(), "en": en.strip(), "type": ""})
    return entities


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_csv_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv_df(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            out = dict(row)
            for key in ("entities", "entities_gold", "entities_model", "entities_eval"):
                if key in out and isinstance(out[key], str):
                    out[key] = json.loads(out[key])
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def build_type_maps(term_all: pd.DataFrame) -> tuple[dict[tuple[str, str], str], dict[str, str], list[dict[str, Any]]]:
    by_pair: dict[tuple[str, str], str] = {}
    zh_types: dict[str, set[str]] = defaultdict(set)
    issues: list[dict[str, Any]] = []
    for row in term_all.to_dict(orient="records"):
        zh = str(row["zh"]).strip()
        en = str(row["en"]).strip()
        etype = str(row["entity_type"]).strip()
        by_pair[(zh, en)] = etype
        zh_types[zh].add(etype)
    by_zh: dict[str, str] = {}
    for zh, types in zh_types.items():
        if len(types) == 1:
            by_zh[zh] = next(iter(types))
        else:
            issues.append({"issue": "ambiguous_zh_type", "zh": zh, "types": sorted(types)})
    return by_pair, by_zh, issues


def fill_entity_types(entities: list[dict[str, str]], by_pair: dict[tuple[str, str], str], by_zh: dict[str, str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for ent in entities:
        zh = ent["zh"]
        en = ent["en"]
        etype = by_pair.get((zh, en), by_zh.get(zh, ""))
        result.append({"zh": zh, "en": en, "type": etype})
    return result


def unique_entities(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, str]] = []
    for ent in entities:
        key = (ent.get("zh", ""), ent.get("en", ""), ent.get("type", ""))
        if key[0] and key[1] and key not in seen:
            seen.add(key)
            result.append(ent)
    return result


def find_spans(text: str, entities: list[dict[str, str]]) -> list[tuple[int, int, dict[str, str]]]:
    occupied = [False] * len(text)
    spans: list[tuple[int, int, dict[str, str]]] = []
    for ent in sorted(unique_entities(entities), key=lambda item: len(item["zh"]), reverse=True):
        start = 0
        while True:
            idx = text.find(ent["zh"], start)
            if idx < 0:
                break
            end = idx + len(ent["zh"])
            if not any(occupied[idx:end]):
                spans.append((idx, end, ent))
                for pos in range(idx, end):
                    occupied[pos] = True
            start = idx + 1
    return sorted(spans, key=lambda item: item[0])


def inject_entities(text: str, entities: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    spans = find_spans(text, entities)
    if not spans:
        return text, []
    chunks: list[str] = []
    detected: list[dict[str, str]] = []
    cursor = 0
    for start, end, ent in spans:
        chunks.append(text[cursor:start])
        chunks.append(f"[{ent['type']}] {ent['en']} [/{ent['type']}]")
        detected.append(ent)
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks), detected


def prepared_rows(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plain_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        entities_model = json.loads(row["entities_model"])
        injected_source, injected_entities = inject_entities(str(row["zh"]), entities_model)
        base = {
            "id": row["id"],
            "split": row["split"],
            "source": row["zh"],
            "source_entity": injected_source,
            "target": row["en"],
            "text_type": row.get("text_type", ""),
            "entities": json.loads(row["entities_eval"]),
            "entities_eval": json.loads(row["entities_eval"]),
            "entities_model": injected_entities,
            "entities_gold": json.loads(row["entities_gold"]),
            "source_status": row.get("source_status", ""),
        }
        plain_rows.append({**base, "source_for_model": row["zh"]})
        entity_rows.append({**base, "source_for_model": injected_source})
    return plain_rows, entity_rows


def choose_removed_id(a: dict[str, Any], b: dict[str, Any]) -> str:
    pa = STRICT_KEEP_PRIORITY[str(a["split"])]
    pb = STRICT_KEEP_PRIORITY[str(b["split"])]
    if pa == pb:
        return ""
    return str(a["id"]) if pa < pb else str(b["id"])


def detect_near_duplicates(all_rows: pd.DataFrame) -> tuple[set[str], list[dict[str, Any]]]:
    rows = all_rows[["id", "split", "zh", "en"]].to_dict(orient="records")
    audit: list[dict[str, Any]] = []
    remove_ids: set[str] = set()
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if a["split"] == b["split"]:
                continue
            zh_a, zh_b = norm_zh(a["zh"]), norm_zh(b["zh"])
            en_a, en_b = norm_en(a["en"]), norm_en(b["en"])
            zh_score = SequenceMatcher(None, zh_a, zh_b).ratio() if min(len(zh_a), len(zh_b)) >= 6 else 0.0
            en_score = SequenceMatcher(None, en_a, en_b).ratio() if min(len(en_a), len(en_b)) >= 25 else 0.0
            reason = ""
            if en_a and en_a == en_b:
                reason = "normalized_en_duplicate"
            elif zh_score >= 0.88:
                reason = "cross_split_near_zh"
            elif en_score >= 0.92:
                reason = "cross_split_near_en"
            if not reason:
                continue
            removed = choose_removed_id(a, b)
            if removed:
                remove_ids.add(removed)
            audit.append(
                {
                    "reason": reason,
                    "id_a": a["id"],
                    "split_a": a["split"],
                    "zh_a": a["zh"],
                    "en_a": a["en"],
                    "id_b": b["id"],
                    "split_b": b["split"],
                    "zh_b": b["zh"],
                    "en_b": b["en"],
                    "zh_similarity": round(zh_score, 4),
                    "en_similarity": round(en_score, 4),
                    "removed_id_for_strict": removed,
                }
            )
    return remove_ids, audit


def make_entity_rows(frames: dict[str, pd.DataFrame], term_all: pd.DataFrame, term_train: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    by_pair, by_zh, type_issues = build_type_maps(term_all)
    train_terms = set(term_train["zh"].astype(str))
    out_frames: dict[str, pd.DataFrame] = {}
    entity_audit_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"type_map_issues": type_issues, "splits": {}}

    for split, df in frames.items():
        rows: list[dict[str, Any]] = []
        counters = Counter()
        for row in df.to_dict(orient="records"):
            gold = fill_entity_types(parse_entity_string(row["entities"]), by_pair, by_zh)
            model = [
                ent
                for ent in gold
                if ent["zh"] in train_terms and ent["type"] and ent["zh"] in str(row["zh"])
            ]
            eval_entities = [
                ent
                for ent in gold
                if ent["type"] and ent["zh"] in str(row["zh"]) and contains_en(str(row["en"]), ent["en"])
            ]
            counters["gold_occurrences"] += len(gold)
            counters["model_occurrences"] += len(model)
            counters["eval_occurrences"] += len(eval_entities)
            if model:
                counters["rows_with_model_entities"] += 1
            if len(model) == len(gold):
                counters["rows_all_gold_are_model_entities"] += 1
            if len(eval_entities) == len(gold):
                counters["rows_all_gold_are_eval_entities"] += 1
            if len(eval_entities) != len(gold) or len(model) != len(gold):
                entity_audit_rows.append(
                    {
                        "id": row["id"],
                        "split": split,
                        "zh": row["zh"],
                        "en": row["en"],
                        "gold_count": len(gold),
                        "model_count": len(model),
                        "eval_count": len(eval_entities),
                        "entities_gold": dump_json(gold),
                        "entities_model": dump_json(model),
                        "entities_eval": dump_json(eval_entities),
                    }
                )
            out = dict(row)
            out["entities_gold"] = dump_json(gold)
            out["entities_model"] = dump_json(model)
            out["entities_eval"] = dump_json(eval_entities)
            out["entities"] = out["entities_eval"]
            out["term_count_gold"] = len(gold)
            out["term_count_model"] = len(model)
            out["term_count_eval"] = len(eval_entities)
            rows.append(out)
        summary["splits"][split] = dict(counters)
        out_frames[split] = pd.DataFrame(rows)

    summary["entity_audit_rows"] = entity_audit_rows
    return out_frames, summary


def write_data_tree(base: Path, frames: dict[str, pd.DataFrame]) -> None:
    ensure_dir(base)
    for split, df in frames.items():
        write_csv_df(base / f"{split}.csv", df)
        write_jsonl(base / f"{split}.jsonl", df.to_dict(orient="records"))


def write_nllb_parallel(base: Path, frames: dict[str, pd.DataFrame]) -> None:
    ensure_dir(base)
    for split, df in frames.items():
        (base / f"{split}.zho_Hans").write_text("\n".join(df["zh"].astype(str)) + "\n", encoding="utf-8")
        (base / f"{split}.eng_Latn").write_text("\n".join(df["en"].astype(str)) + "\n", encoding="utf-8")


def write_prepared(base: Path, frames: dict[str, pd.DataFrame]) -> None:
    ensure_dir(base)
    for split, df in frames.items():
        plain, entity = prepared_rows(df)
        write_jsonl(base / f"{split}_plain.jsonl", plain)
        write_jsonl(base / f"{split}_entity.jsonl", entity)


def write_eval_terminology(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    evidence: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for split, df in frames.items():
        for row in df.to_dict(orient="records"):
            for ent in json.loads(row["entities_eval"]):
                key = (ent["zh"], ent["en"], ent["type"])
                counts[key][split] += 1
                evidence[key].append(row["id"])
    rows = []
    for idx, (key, counter) in enumerate(sorted(counts.items()), start=1):
        zh, en, etype = key
        rows.append(
            {
                "term_id": f"TERM_EVAL_{idx:04d}",
                "zh": zh,
                "en": en,
                "entity_type": etype,
                "train_count": counter["train"],
                "dev_count": counter["dev"],
                "test_count": counter["test"],
                "occurrence_count": sum(counter.values()),
                "evidence_ids": ", ".join(evidence[key]),
            }
        )
    write_csv_df(path, pd.DataFrame(rows))


def write_bio_audit(path: Path, source_root: Path, frames: dict[str, pd.DataFrame]) -> None:
    rows: list[dict[str, Any]] = []
    bio_root = source_root / "bio_experiment_csv_package"
    for split, df in frames.items():
        spans = read_csv_df(bio_root / f"bio_spans_{split}.csv").set_index("id")
        for row in df.to_dict(orient="records"):
            span_count = int(spans.loc[row["id"], "span_count"]) if row["id"] in spans.index else 0
            gold_count = int(row["term_count_gold"])
            if span_count != gold_count:
                rows.append(
                    {
                        "id": row["id"],
                        "split": split,
                        "zh": row["zh"],
                        "term_count_gold": gold_count,
                        "bio_span_count": span_count,
                        "gap": gold_count - span_count,
                        "reason": "BIO is non-overlapping; nested, discontinuous, or non-exact entities are not fully represented.",
                        "entities_gold": row["entities_gold"],
                        "bio_spans_json": spans.loc[row["id"], "spans_json"] if row["id"] in spans.index else "",
                    }
                )
    write_csv_df(path, pd.DataFrame(rows))


def write_report_files(ready_root: Path, source_root: Path, full_frames: dict[str, pd.DataFrame], strict_frames: dict[str, pd.DataFrame], entity_summary: dict[str, Any], duplicate_audit: list[dict[str, Any]], removed_ids: set[str]) -> None:
    reports = ready_root / "reports"
    ensure_dir(reports)
    all_full = pd.concat(full_frames.values(), ignore_index=True)
    all_strict = pd.concat(strict_frames.values(), ignore_index=True)
    term_all = read_csv_df(source_root / "terminology_all.csv")
    term_train = read_csv_df(source_root / "terminology_train.csv")
    validation = {
        "source_root": str(source_root),
        "ready_root": str(ready_root),
        "data_full_counts": {split: int(len(df)) for split, df in full_frames.items()},
        "data_strict_counts": {split: int(len(df)) for split, df in strict_frames.items()},
        "strict_removed_ids": sorted(removed_ids),
        "strict_removed_count": len(removed_ids),
        "missing_values_full": {col: int(all_full[col].isna().sum()) for col in all_full.columns if int(all_full[col].isna().sum())},
        "duplicate_id_full": int(all_full.duplicated("id").sum()),
        "duplicate_zh_full": int(all_full.duplicated("zh").sum()),
        "duplicate_en_full": int(all_full.duplicated("en").sum()),
        "duplicate_zh_norm_full": int(all_full["zh"].map(norm_zh).duplicated().sum()),
        "duplicate_en_norm_full": int(all_full["en"].map(norm_en).duplicated().sum()),
        "duplicate_id_strict": int(all_strict.duplicated("id").sum()),
        "duplicate_zh_strict": int(all_strict.duplicated("zh").sum()),
        "duplicate_en_strict": int(all_strict.duplicated("en").sum()),
        "duplicate_zh_norm_strict": int(all_strict["zh"].map(norm_zh).duplicated().sum()),
        "duplicate_en_norm_strict": int(all_strict["en"].map(norm_en).duplicated().sum()),
        "terminology_all_rows": int(len(term_all)),
        "terminology_train_rows": int(len(term_train)),
        "single_char_terms_all": int((term_all["zh"].astype(str).str.len() == 1).sum()),
        "single_char_terms_train": int((term_train["zh"].astype(str).str.len() == 1).sum()),
        "entity_summary": entity_summary["splits"],
        "type_map_issues": entity_summary["type_map_issues"],
    }
    (reports / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    leakage = {
        "policy": "Main experiments use terminology_train.csv and entities_model only. terminology_all.csv is kept only for audit/gold metadata.",
        "test_only_terms": int(((term_all["train_count"].fillna(0) == 0) & (term_all["dev_count"].fillna(0) == 0) & (term_all["test_count"].fillna(0) > 0)).sum()),
        "no_train_terms": int(((term_all["train_count"].fillna(0) == 0) & ((term_all["dev_count"].fillna(0) > 0) | (term_all["test_count"].fillna(0) > 0))).sum()),
        "train_table_contains_dev_evidence": int(term_train["evidence_ids"].astype(str).str.contains("TCM_DEV_", regex=False).sum()),
        "train_table_contains_test_evidence": int(term_train["evidence_ids"].astype(str).str.contains("TCM_TEST_", regex=False).sum()),
        "model_input_uses_test_only_terms": False,
    }
    (reports / "leakage_audit.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv_df(reports / "near_duplicate_audit.csv", pd.DataFrame(duplicate_audit))
    write_csv_df(reports / "entity_audit.csv", pd.DataFrame(entity_summary["entity_audit_rows"]))
    write_bio_audit(reports / "entity_bio_audit.csv", source_root, full_frames)

    readme = [
        "# YYQEI 2026-05-08 NLLB-ready Data",
        "",
        "This folder was generated from `source_materials` for no-leak TCM Chinese-to-English MT experiments.",
        "",
        "## Main Policy",
        "- `data/` is the strict default split used by the experiment script.",
        "- `data_full/` preserves the original 1000/250/250 split.",
        "- `data_strict/` removes cross-split exact or high-similarity near duplicates.",
        "- Model entity tags use only `terminology/terminology_train.csv` through `entities_model`.",
        "- TA/TCR scoring uses `entities_eval`, which requires exact Chinese source and English reference evidence.",
        "- `terminology_all.csv` is audit/gold metadata only and must not be used for test-time prompting.",
        "- The experiment script disables Terminology-suffix post-processing and reports raw outputs as primary results.",
        "",
        "## Counts",
        f"- data_full: {validation['data_full_counts']}",
        f"- data_strict/default data: {validation['data_strict_counts']}",
        f"- strict removed IDs: {len(removed_ids)}",
        f"- terminology_train rows: {len(term_train)}",
        f"- terminology_all rows: {len(term_all)}",
        "",
        "## Run",
        "From the package root:",
        "",
        "```powershell",
        "& 'D:\\Anaconda\\Scripts\\conda.exe' run -n YYQEI python scripts\\validate_ready_package.py --root .\\YYQEI_20260508_NLLB_ready",
        "& 'D:\\Anaconda\\Scripts\\conda.exe' run -n YYQEI python scripts\\nllb_tcm_experiment.py check-env --root .\\YYQEI_20260508_NLLB_ready --run-dir .\\runs\\nllb_20260508",
        "```",
    ]
    (reports / "README_NLLB_READY.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> None:
    root = package_root()
    source_root = root / "source_materials"
    ready_root = root / "YYQEI_20260508_NLLB_ready"
    if ready_root.exists():
        shutil.rmtree(ready_root)
    ensure_dir(ready_root)

    raw_frames = {split: read_csv_df(source_root / "data" / f"{split}.csv") for split in SPLITS}
    term_all = read_csv_df(source_root / "terminology_all.csv")
    term_train = read_csv_df(source_root / "terminology_train.csv")
    full_frames, entity_summary = make_entity_rows(raw_frames, term_all, term_train)

    all_full = pd.concat(full_frames.values(), ignore_index=True)
    removed_ids, duplicate_audit = detect_near_duplicates(all_full)
    strict_frames = {
        split: df[~df["id"].astype(str).isin(removed_ids)].reset_index(drop=True)
        for split, df in full_frames.items()
    }

    write_data_tree(ready_root / "data_full", full_frames)
    write_data_tree(ready_root / "data_strict", strict_frames)
    write_data_tree(ready_root / "data", strict_frames)

    ensure_dir(ready_root / "terminology")
    shutil.copy2(source_root / "terminology_train.csv", ready_root / "terminology" / "terminology_train.csv")
    shutil.copy2(source_root / "terminology_all.csv", ready_root / "terminology" / "terminology_all.csv")
    write_eval_terminology(ready_root / "terminology" / "terminology_eval_gold.csv", strict_frames)

    ensure_dir(ready_root / "bio")
    shutil.copy2(source_root / "Bio.xlsx", ready_root / "bio" / "Bio.xlsx")
    shutil.copytree(source_root / "bio_experiment_csv_package", ready_root / "bio" / "bio_experiment_csv_package")

    write_nllb_parallel(ready_root / "nllb_parallel", strict_frames)
    write_prepared(ready_root / "prepared", strict_frames)
    write_report_files(ready_root, source_root, full_frames, strict_frames, entity_summary, duplicate_audit, removed_ids)

    print(f"[build] wrote {ready_root}")
    print(f"[build] data_full counts: { {split: len(df) for split, df in full_frames.items()} }")
    print(f"[build] data_strict counts: { {split: len(df) for split, df in strict_frames.items()} }")
    print(f"[build] strict removed ids: {sorted(removed_ids)}")


if __name__ == "__main__":
    main()
