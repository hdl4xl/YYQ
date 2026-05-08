from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DATA_COLUMNS = [
    "id",
    "split",
    "zh",
    "en",
    "text_type",
    "term_count",
    "entities",
    "source_status",
    "note",
]

SHEET_NAMES = {"train": "Train", "dev": "dev", "test": "Test"}


ROW_CORRECTIONS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "TCM_TRAIN_0214": {"en": [("cireculation", "circulation")]},
    "TCM_TRAIN_0351": {
        "en": [
            ("transporation", "transportation"),
            ("distentim", "distention"),
            ("in ternal", "internal"),
        ]
    },
    "TCM_TRAIN_0428": {"en": [("etc.in", "etc. in")]},
    "TCM_TRAIN_0429": {"en": [("clinicai", "clinical"), ("etc.in", "etc. in")]},
    "TCM_TRAIN_0457": {
        "en": [
            ("upward and...; downward", "upward and downward"),
            ("orgnas", "organs"),
        ]
    },
    "TCM_TRAIN_0500": {"en": [("ofter", "often")]},
    "TCM_TRAIN_0850": {"en": [("monifestations", "manifestations")]},
    "TCM_TRAIN_0988": {
        "en": [("lo stagnation", "to stagnation"), ("insuficiency", "insufficiency")]
    },
    "TCM_DEV_0015": {"en": [("wood, fire earth", "wood, fire, earth")]},
    "TCM_DEV_0041": {"en": [("consitions", "conditions")]},
    "TCM_DEV_0240": {"en": [("manifestated", "manifested")]},
    "TCM_TEST_0027": {"zh": [("反之.肾精不足", "反之，肾精不足")]},
    "TCM_TEST_0038": {"zh": [("脾、胃.肝、胆", "脾、胃、肝、胆")]},
    "TCM_TEST_0083": {"zh": [("易致肿疡.火邪", "易致肿疡。火邪")]},
    "TCM_TEST_0142": {"zh": [("膏粱厚味.助火", "膏粱厚味，助火")]},
    "TCM_TEST_0164": {"zh": [("风痰.或", "风痰，或")]},
    "TCM_TEST_0173": {"zh": [("关脉.尺脉", "关脉、尺脉")]},
    "TCM_TEST_0181": {
        "zh": [("气盛血.脉道", "气盛血，脉道"), ("故脉体阔大.浮大", "故脉体阔大，浮大")]
    },
}

KNOWN_BAD_EN = re.compile(
    r"(iiinvolves|iii+involves|llllustre|llllustrelessness|tttthe|pabient|mnaed|"
    r"warim|obstuction|hungs|spleer|timbs|clinicai|orgnas|ofter|body luid|"
    r"oocurs|transporation|per minuie|consitions|cireculation|monifestations|"
    r"bleading|polyorexia|insuficiency|seriouse|lo stagnation|manifestated|"
    r"distentim|in ternal)",
    re.IGNORECASE,
)


def parse_entities(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        raw = value
    else:
        raw = json.loads(value)
    return [
        {"zh": str(item["zh"]), "en": str(item["en"]), "type": str(item["type"])}
        for item in raw
    ]


def dump_entities(entities: list[dict[str, str]]) -> str:
    return json.dumps(entities, ensure_ascii=False, separators=(",", ":"))


def norm_zh(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKC", str(text))
        if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", ch)
    )


def norm_en(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", str(text)).lower())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_split_frames(input_materials: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in ["train", "dev", "test"]:
        path = input_materials / "Data" / f"{split}.xlsx"
        df = pd.read_excel(path)
        df = df[DATA_COLUMNS].copy()
        frames[split] = df
    return frames


def apply_row_corrections(frames: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    logs: list[dict[str, str]] = []
    for split, df in frames.items():
        for idx, row in df.iterrows():
            row_id = str(row["id"])
            if row_id not in ROW_CORRECTIONS:
                continue
            for field, replacements in ROW_CORRECTIONS[row_id].items():
                value = str(df.at[idx, field])
                before = value
                for old, new in replacements:
                    value = value.replace(old, new)
                if value != before:
                    df.at[idx, field] = value
                    logs.append(
                        {
                            "id": row_id,
                            "split": split,
                            "field": field,
                            "before": before,
                            "after": value,
                        }
                    )
    return logs


def canonicalize_entities(frames: dict[str, pd.DataFrame]) -> None:
    for df in frames.values():
        for idx, row in df.iterrows():
            entities = parse_entities(row["entities"])
            df.at[idx, "entities"] = dump_entities(entities)
            df.at[idx, "term_count"] = len(entities)


def write_csv_utf8_sig(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataframe_to_jsonl_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        out = dict(row)
        out["term_count"] = int(out["term_count"])
        out["entities"] = parse_entities(out["entities"])
        rows.append(out)
    return rows


def write_split_tables(frames: dict[str, pd.DataFrame], materials_out: Path) -> None:
    data_out = materials_out / "Data"
    ensure_dir(data_out)
    for split, df in frames.items():
        sheet_name = SHEET_NAMES[split]
        with pd.ExcelWriter(data_out / f"{split}.xlsx", engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        write_csv_utf8_sig(data_out / f"{split}.csv", df)
        write_jsonl(data_out / f"{split}.jsonl", dataframe_to_jsonl_rows(df))


def entity_map_for_row(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    by_zh: dict[str, dict[str, str]] = {}
    for ent in entities:
        if ent["zh"]:
            by_zh[ent["zh"]] = ent
    return sorted(by_zh.values(), key=lambda item: len(item["zh"]), reverse=True)


def find_spans(text: str, entities: list[dict[str, str]]) -> list[dict[str, Any]]:
    occupied = [False] * len(text)
    spans: list[dict[str, Any]] = []
    for ent in entity_map_for_row(entities):
        term = ent["zh"]
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            end = idx + len(term)
            if not any(occupied[idx:end]):
                spans.append({"start": idx, "end": end, "text": term, "type": ent["type"]})
                for pos in range(idx, end):
                    occupied[pos] = True
            start = idx + 1
    return sorted(spans, key=lambda item: (item["start"], item["end"]))


def inject_entity_tags(text: str, entities: list[dict[str, str]]) -> str:
    spans = find_spans(text, entities)
    ent_by_key = {(ent["zh"], ent["type"]): ent for ent in entities}
    parts: list[str] = []
    cursor = 0
    for span in spans:
        ent = ent_by_key.get((span["text"], span["type"]))
        if not ent:
            continue
        parts.append(text[cursor : span["start"]])
        parts.append(f"[{ent['type']}] {ent['en']} [/{ent['type']}]")
        cursor = span["end"]
    parts.append(text[cursor:])
    return "".join(parts)


def build_bio(frames: dict[str, pd.DataFrame], bio_out: Path) -> None:
    span_rows: list[dict[str, Any]] = []
    char_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    preview_ids: set[str] = set()

    for split, df in frames.items():
        for _, row in df.iterrows():
            row_id = str(row["id"])
            zh = str(row["zh"])
            entities = parse_entities(row["entities"])
            spans = find_spans(zh, entities)
            span_rows.append(
                {
                    "id": row_id,
                    "split": split,
                    "zh": zh,
                    "spans_json": json.dumps(spans, ensure_ascii=False, separators=(",", ":")),
                    "span_count": len(spans),
                    "zh_len": len(zh),
                    "text_type": row["text_type"],
                }
            )

            labels = ["O"] * len(zh)
            types = [""] * len(zh)
            for span in spans:
                for pos in range(span["start"], span["end"]):
                    labels[pos] = ("B-" if pos == span["start"] else "I-") + span["type"]
                    types[pos] = span["type"]

            for pos, ch in enumerate(zh):
                item = {
                    "id": row_id,
                    "split": split,
                    "token_index": pos,
                    "char": ch,
                    "bio_label": labels[pos],
                    "entity_type": types[pos],
                }
                char_rows.append(item)
                if split == "train" and len(preview_ids) < 40:
                    preview_ids.add(row_id)
                if row_id in preview_ids:
                    preview_rows.append(item)

    ensure_dir(bio_out.parent)
    with pd.ExcelWriter(bio_out, engine="openpyxl") as writer:
        pd.DataFrame(span_rows).to_excel(writer, sheet_name="BIO_Spans", index=False)
        pd.DataFrame(char_rows).to_excel(writer, sheet_name="BIO_Char_Full", index=False)
        pd.DataFrame(preview_rows).to_excel(writer, sheet_name="BIO_Char_Preview", index=False)


def write_generated_materials(
    frames: dict[str, pd.DataFrame], input_materials: Path, generated_out: Path
) -> None:
    data_out = generated_out / "data"
    ensure_dir(data_out)
    for split, df in frames.items():
        write_csv_utf8_sig(data_out / f"{split}.csv", df)

    term_all = pd.read_excel(input_materials / "terminology.xlsx", sheet_name="terminology_all")
    term_out = generated_out / "terminology"
    ensure_dir(term_out)
    term_cols = ["zh", "en", "entity_type", "aliases"]
    term_all[term_cols].to_csv(
        term_out / "terminology.csv", index=False, encoding="utf-8-sig", lineterminator="\n"
    )


def write_nllb_files(frames: dict[str, pd.DataFrame], output_root: Path) -> None:
    parallel_dir = output_root / "nllb_parallel"
    prepared_dir = output_root / "runs" / "nllb_real_data" / "prepared"
    ensure_dir(parallel_dir)
    ensure_dir(prepared_dir)

    for split, df in frames.items():
        src_lines: list[str] = []
        tgt_lines: list[str] = []
        plain_rows: list[dict[str, Any]] = []
        entity_rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            entities = parse_entities(row["entities"])
            source = str(row["zh"])
            target = str(row["en"])
            source_entity = inject_entity_tags(source, entities)
            base = {
                "id": row["id"],
                "split": split,
                "source": source,
                "source_entity": source_entity,
                "target": target,
                "text_type": row.get("text_type", ""),
                "entities": entities,
                "source_status": row.get("source_status", ""),
            }
            plain_rows.append({**base, "source_for_model": source})
            entity_rows.append({**base, "source_for_model": source_entity})
            src_lines.append(source)
            tgt_lines.append(target)

        (parallel_dir / f"{split}.zho_Hans").write_text(
            "\n".join(src_lines) + "\n", encoding="utf-8"
        )
        (parallel_dir / f"{split}.eng_Latn").write_text(
            "\n".join(tgt_lines) + "\n", encoding="utf-8"
        )
        write_jsonl(prepared_dir / f"{split}_plain.jsonl", plain_rows)
        write_jsonl(prepared_dir / f"{split}_entity.jsonl", entity_rows)


def validate(frames: dict[str, pd.DataFrame], materials_out: Path) -> dict[str, Any]:
    all_df = pd.concat(frames.values(), ignore_index=True)
    report: dict[str, Any] = {}
    report["split_counts"] = {split: int(len(df)) for split, df in frames.items()}
    report["total_rows"] = int(len(all_df))
    report["missing_values"] = {
        col: int(all_df[col].isna().sum()) for col in DATA_COLUMNS if int(all_df[col].isna().sum())
    }
    report["duplicate_id_rows"] = int(all_df.duplicated("id").sum())
    report["duplicate_zh_rows"] = int(all_df.duplicated("zh").sum())
    report["duplicate_en_rows"] = int(all_df.duplicated("en").sum())
    report["duplicate_pair_rows"] = int(all_df.duplicated(["zh", "en"]).sum())
    report["duplicate_zh_norm_rows"] = int(all_df["zh"].map(norm_zh).duplicated().sum())
    report["duplicate_en_norm_rows"] = int(all_df["en"].map(norm_en).duplicated().sum())

    entity_rows: list[dict[str, str]] = []
    entity_errors: list[str] = []
    for _, row in all_df.iterrows():
        entities = parse_entities(row["entities"])
        if int(row["term_count"]) != len(entities):
            entity_errors.append(f"{row['id']}: term_count mismatch")
        for ent in entities:
            if ent["zh"] not in str(row["zh"]):
                entity_errors.append(f"{row['id']}: entity not in zh: {ent['zh']}")
            entity_rows.append(ent)
    report["entity_errors"] = entity_errors
    report["entity_mentions"] = len(entity_rows)
    report["unique_terms"] = len({ent["zh"] for ent in entity_rows})
    report["known_bad_en_rows"] = [
        str(row["id"]) for _, row in all_df.iterrows() if KNOWN_BAD_EN.search(str(row["en"]))
    ]
    report["ascii_period_zh_rows"] = [
        str(row["id"]) for _, row in all_df.iterrows() if "." in str(row["zh"])
    ]

    term_all = pd.read_excel(materials_out / "terminology.xlsx", sheet_name="terminology_all")
    term_zh = set(term_all["zh"].astype(str))
    entity_zh = {ent["zh"] for ent in entity_rows}
    report["term_coverage_missing"] = sorted(entity_zh - term_zh)
    report["term_extra_not_in_entities"] = sorted(term_zh - entity_zh)

    spans = pd.read_excel(materials_out / "bio.xlsx", sheet_name="BIO_Spans")
    chars = pd.read_excel(materials_out / "bio.xlsx", sheet_name="BIO_Char_Full")
    report["bio_spans_rows"] = int(len(spans))
    report["bio_char_full_rows"] = int(len(chars))
    report["bio_ids_missing"] = sorted(set(all_df["id"].astype(str)) - set(spans["id"].astype(str)))
    report["bio_char_ids_missing"] = sorted(
        set(all_df["id"].astype(str)) - set(chars["id"].astype(str))
    )

    source_by_id = dict(zip(all_df["id"].astype(str), all_df["zh"].astype(str)))
    span_errors: list[str] = []
    for _, row in spans.iterrows():
        zh = str(row["zh"])
        if source_by_id.get(str(row["id"])) != zh:
            span_errors.append(f"{row['id']}: zh mismatch")
        for span in json.loads(row["spans_json"]):
            if zh[int(span["start"]) : int(span["end"])] != span["text"]:
                span_errors.append(f"{row['id']}: span mismatch {span}")
                break
    report["bio_span_errors"] = span_errors

    char_errors: list[str] = []
    for row_id, group in chars.groupby("id", sort=False):
        zh = source_by_id.get(str(row_id), "")
        recon = "".join(group.sort_values("token_index")["char"].astype(str).tolist())
        if recon != zh:
            char_errors.append(str(row_id))
    report["bio_char_reconstruction_errors"] = char_errors
    report["text_type_counts"] = dict(Counter(all_df["text_type"].astype(str)))
    return report


def write_report(output_root: Path, report: dict[str, Any], correction_log: list[dict[str, str]]) -> None:
    report_dir = output_root / "reports"
    ensure_dir(report_dir)
    (report_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (report_dir / "correction_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["id", "split", "field", "before", "after"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(correction_log)

    summary = [
        "# NLLB-ready YYQEI Data",
        "",
        f"- total_rows: {report['total_rows']}",
        f"- split_counts: {report['split_counts']}",
        f"- duplicate_zh_norm_rows: {report['duplicate_zh_norm_rows']}",
        f"- duplicate_en_norm_rows: {report['duplicate_en_norm_rows']}",
        f"- known_bad_en_rows: {report['known_bad_en_rows']}",
        f"- ascii_period_zh_rows: {report['ascii_period_zh_rows']}",
        f"- term_coverage_missing: {report['term_coverage_missing']}",
        f"- bio_span_errors: {len(report['bio_span_errors'])}",
        f"- bio_char_reconstruction_errors: {len(report['bio_char_reconstruction_errors'])}",
        f"- corrections_applied: {len(correction_log)}",
        "",
        "Use with the existing script:",
        "",
        "```powershell",
        (
            "python scripts\\nllb_tcm_experiment.py prepare "
            f"--root {output_root} --run-dir runs\\nllb_real_data"
        ),
        "```",
    ]
    (report_dir / "README_NLLB_READY.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NLLB-ready YYQEI real-data materials.")
    parser.add_argument(
        "--input-materials",
        default=r"D:\Desktop\YYQEI\YYQEI_Materials_cleaned\Materials",
    )
    parser.add_argument("--output-root", default=r"D:\Desktop\YYQEI\YYQEI_NLLB_ready")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_materials = Path(args.input_materials)
    output_root = Path(args.output_root)
    materials_out = output_root / "Materials"

    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    if output_root.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output_root}. Use --force to rebuild.")

    frames = read_split_frames(input_materials)
    correction_log = apply_row_corrections(frames)
    canonicalize_entities(frames)

    ensure_dir(materials_out)
    write_split_tables(frames, materials_out)
    shutil.copy2(input_materials / "terminology.xlsx", materials_out / "terminology.xlsx")
    build_bio(frames, materials_out / "bio.xlsx")
    write_generated_materials(frames, input_materials, output_root / "generated_materials")
    write_nllb_files(frames, output_root)

    report = validate(frames, materials_out)
    write_report(output_root, report, correction_log)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
