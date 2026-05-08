from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


SPLITS = ("train", "dev", "test")


def norm_zh(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff" or ch.isascii() and ch.isalnum())


def norm_en(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"[FAIL] {message}")


def ok(message: str) -> None:
    print(f"[PASS] {message}")


def read_csv(path: Path, errors: list[str]) -> pd.DataFrame:
    if not path.exists():
        fail(errors, f"missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_json_entities(value: Any, errors: list[str], row_id: str, field: str) -> list[dict[str, str]]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception as exc:  # noqa: BLE001 - validator reports all parse failures.
        fail(errors, f"{row_id}: {field} is not valid JSON: {exc}")
        return []
    if not isinstance(parsed, list):
        fail(errors, f"{row_id}: {field} is not a list")
        return []
    result: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            fail(errors, f"{row_id}: {field} contains non-object entity")
            continue
        zh = str(item.get("zh", ""))
        en = str(item.get("en", ""))
        etype = str(item.get("type", ""))
        if not zh or not en or not etype:
            fail(errors, f"{row_id}: {field} has incomplete entity {item}")
        result.append({"zh": zh, "en": en, "type": etype})
    return result


def validate_data(root: Path, errors: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    required_columns = {
        "id",
        "split",
        "zh",
        "en",
        "text_type",
        "term_count",
        "entities",
        "entities_gold",
        "entities_model",
        "entities_eval",
        "term_count_gold",
        "term_count_model",
        "term_count_eval",
    }
    for split in SPLITS:
        df = read_csv(root / "data" / f"{split}.csv", errors)
        frames[split] = df
        if df.empty:
            continue
        missing = required_columns - set(df.columns)
        if missing:
            fail(errors, f"{split}: missing columns {sorted(missing)}")
        if int((df["split"] != split).sum()):
            fail(errors, f"{split}: split column mismatch")
        for row in df.to_dict(orient="records"):
            row_id = str(row["id"])
            gold = load_json_entities(row.get("entities_gold", "[]"), errors, row_id, "entities_gold")
            model = load_json_entities(row.get("entities_model", "[]"), errors, row_id, "entities_model")
            eval_entities = load_json_entities(row.get("entities_eval", "[]"), errors, row_id, "entities_eval")
            if len(gold) != int(row.get("term_count_gold", -1)):
                fail(errors, f"{row_id}: term_count_gold mismatch")
            if len(model) != int(row.get("term_count_model", -1)):
                fail(errors, f"{row_id}: term_count_model mismatch")
            if len(eval_entities) != int(row.get("term_count_eval", -1)):
                fail(errors, f"{row_id}: term_count_eval mismatch")
            for ent in model:
                if ent["zh"] not in str(row["zh"]):
                    fail(errors, f"{row_id}: model entity not in source: {ent['zh']}")
            for ent in eval_entities:
                if ent["zh"] not in str(row["zh"]):
                    fail(errors, f"{row_id}: eval entity not in source: {ent['zh']}")
                if ent["en"].lower() not in str(row["en"]).lower():
                    fail(errors, f"{row_id}: eval entity not in reference: {ent['en']}")
        ok(f"{split}: rows={len(df)}")
    all_df = pd.concat(frames.values(), ignore_index=True)
    if int(all_df.duplicated("id").sum()):
        fail(errors, "duplicate ids in data")
    if int(all_df.duplicated("zh").sum()):
        fail(errors, "duplicate zh in data")
    if int(all_df.duplicated("en").sum()):
        fail(errors, "duplicate en in data")
    if int(all_df["zh"].map(norm_zh).duplicated().sum()):
        fail(errors, "normalized zh duplicates in data")
    if int(all_df["en"].map(norm_en).duplicated().sum()):
        fail(errors, "normalized en duplicates in data")
    return frames


def validate_terminology(root: Path, frames: dict[str, pd.DataFrame], errors: list[str]) -> None:
    term_train = read_csv(root / "terminology" / "terminology_train.csv", errors)
    term_all = read_csv(root / "terminology" / "terminology_all.csv", errors)
    term_eval = read_csv(root / "terminology" / "terminology_eval_gold.csv", errors)
    if term_train.empty or term_all.empty:
        return
    if int(term_train["evidence_ids"].astype(str).str.contains("TCM_DEV_", regex=False).sum()):
        fail(errors, "terminology_train contains dev evidence")
    if int(term_train["evidence_ids"].astype(str).str.contains("TCM_TEST_", regex=False).sum()):
        fail(errors, "terminology_train contains test evidence")
    train_terms = set(term_train["zh"].astype(str))
    for split, df in frames.items():
        for row in df.to_dict(orient="records"):
            for ent in load_json_entities(row["entities_model"], errors, str(row["id"]), "entities_model"):
                if ent["zh"] not in train_terms:
                    fail(errors, f"{row['id']}: model entity not in terminology_train: {ent['zh']}")
    ok(f"terminology_train rows={len(term_train)}, terminology_all rows={len(term_all)}, terminology_eval rows={len(term_eval)}")


def validate_prepared(root: Path, frames: dict[str, pd.DataFrame], errors: list[str]) -> None:
    for split in SPLITS:
        expected = len(frames[split])
        for kind in ("plain", "entity"):
            path = root / "prepared" / f"{split}_{kind}.jsonl"
            if not path.exists():
                fail(errors, f"missing prepared file: {path}")
                continue
            count = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
            if count != expected:
                fail(errors, f"{path.name}: expected {expected}, got {count}")
        src = root / "nllb_parallel" / f"{split}.zho_Hans"
        tgt = root / "nllb_parallel" / f"{split}.eng_Latn"
        if not src.exists() or not tgt.exists():
            fail(errors, f"missing nllb_parallel files for {split}")
            continue
        src_count = sum(1 for line in src.open("r", encoding="utf-8") if line.rstrip("\n"))
        tgt_count = sum(1 for line in tgt.open("r", encoding="utf-8") if line.rstrip("\n"))
        if src_count != expected or tgt_count != expected:
            fail(errors, f"{split}: nllb_parallel count mismatch {src_count}/{tgt_count}, expected {expected}")
    ok("prepared and nllb_parallel counts match data")


def validate_reports(root: Path, errors: list[str]) -> None:
    for name in ["validation_report.json", "leakage_audit.json", "near_duplicate_audit.csv", "entity_audit.csv", "entity_bio_audit.csv", "README_NLLB_READY.md"]:
        path = root / "reports" / name
        if not path.exists():
            fail(errors, f"missing report: {path}")
    leakage_path = root / "reports" / "leakage_audit.json"
    if leakage_path.exists():
        leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
        if leakage.get("model_input_uses_test_only_terms") is not False:
            fail(errors, "leakage audit does not confirm no test-only model input")
    ok("reports present")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YYQEI 2026-05-08 NLLB-ready package")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "YYQEI_20260508_NLLB_ready"))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    errors: list[str] = []
    print(f"[validate] root={root}")
    frames = validate_data(root, errors)
    validate_terminology(root, frames, errors)
    validate_prepared(root, frames, errors)
    validate_reports(root, errors)
    if errors:
        print(f"[validate] FAILED with {len(errors)} error(s)")
        sys.exit(1)
    print("[validate] PASS")


if __name__ == "__main__":
    main()
