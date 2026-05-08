from __future__ import annotations

import argparse
import csv
import difflib
import importlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "zho_Hans"
TGT_LANG = "eng_Latn"
DEFAULT_RUN_DIR = "runs/nllb_tcm_experiment"


@dataclass
class Term:
    zh: str
    en: str
    entity_type: str
    aliases: list[str]


@dataclass
class ExperimentConfig:
    model_name: str
    src_lang: str
    tgt_lang: str
    seed: int
    max_source_length: int
    max_target_length: int
    generation_max_new_tokens: int
    num_beams: int
    download_retries: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    epochs: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    limit_train: int | None
    limit_eval: int | None
    data_status: str


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def repo_root_from_args(args: argparse.Namespace) -> Path:
    if getattr(args, "root", None):
        return Path(args.root).resolve()
    return Path(__file__).resolve().parents[1]


def run_dir_from_args(args: argparse.Namespace, root: Path) -> Path:
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    return run_dir.resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                out[field] = value
            writer.writerow(out)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_entities(value: str | list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not value:
        return []
    if isinstance(value, list):
        return [
            {
                "zh": str(item.get("zh", "")),
                "en": str(item.get("en", "")),
                "type": str(item.get("type", item.get("entity_type", ""))),
            }
            for item in value
        ]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [
        {
            "zh": str(item.get("zh", "")),
            "en": str(item.get("en", "")),
            "type": str(item.get("type", item.get("entity_type", ""))),
        }
        for item in parsed
    ]


def load_terms(path: Path) -> list[Term]:
    rows = read_csv(path)
    terms: list[Term] = []
    for row in rows:
        aliases = [a.strip() for a in row.get("aliases", "").split(";") if a.strip()]
        term = Term(
            zh=row["zh"].strip(),
            en=row["en"].strip(),
            entity_type=row["entity_type"].strip(),
            aliases=aliases,
        )
        if term.zh and term.en and term.entity_type:
            terms.append(term)
    return sorted(terms, key=lambda t: len(t.zh), reverse=True)


def find_term_spans(text: str, terms: list[Term]) -> list[tuple[int, int, Term]]:
    occupied = [False] * len(text)
    spans: list[tuple[int, int, Term]] = []
    for term in terms:
        start = 0
        while True:
            idx = text.find(term.zh, start)
            if idx < 0:
                break
            end = idx + len(term.zh)
            if not any(occupied[idx:end]):
                spans.append((idx, end, term))
                for pos in range(idx, end):
                    occupied[pos] = True
            start = idx + 1
    return sorted(spans, key=lambda item: item[0])


def inject_entities(text: str, terms: list[Term]) -> tuple[str, list[dict[str, str]]]:
    spans = find_term_spans(text, terms)
    if not spans:
        return text, []

    chunks: list[str] = []
    entities: list[dict[str, str]] = []
    cursor = 0
    for start, end, term in spans:
        chunks.append(text[cursor:start])
        chunks.append(f"[{term.entity_type}] {term.en} [/{term.entity_type}]")
        entities.append({"zh": term.zh, "en": term.en, "type": term.entity_type})
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks), entities


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"['`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_term(prediction: str, term_en: str, aliases: Iterable[str] = ()) -> bool:
    pred_norm = normalize_for_match(prediction)
    candidates = [term_en, *aliases]
    for cand in candidates:
        cand_norm = normalize_for_match(cand)
        if cand_norm and cand_norm in pred_norm:
            return True
    return False


def unique_entities(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, str]] = []
    for ent in entities:
        key = (ent.get("zh", ""), ent.get("en", ""), ent.get("type", ""))
        if key not in seen and key[0] and key[1]:
            seen.add(key)
            result.append(ent)
    return result


def apply_terminology_normalization(prediction: str, entities: list[dict[str, str]]) -> str:
    """Normalize controlled terms with deterministic fuzzy replacement only.

    This 2026-05-08 package deliberately disables the old "Terminology: ..."
    suffix behavior because it leaks answer strings into TA/TCR evaluation.
    """
    text = prediction.strip()
    for ent in unique_entities(entities):
        term_en = ent["en"]
        text = fuzzy_replace_term(text, term_en)
    return text


def word_spans(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"[A-Za-z][A-Za-z-]*", text))


def fuzzy_replace_term(text: str, term_en: str, threshold: float = 0.70) -> str:
    if contains_term(text, term_en):
        return text
    words = word_spans(text)
    term_words = word_spans(term_en)
    if not words or not term_words:
        return text
    target = normalize_for_match(term_en)
    target_len = len(term_words)
    best: tuple[float, int, int] | None = None
    min_window = max(1, target_len - 2)
    max_window = min(len(words), target_len + 2)
    for window in range(min_window, max_window + 1):
        for start in range(0, len(words) - window + 1):
            end = start + window - 1
            candidate = text[words[start].start() : words[end].end()]
            score = difflib.SequenceMatcher(None, normalize_for_match(candidate), target).ratio()
            if best is None or score > best[0]:
                best = (score, words[start].start(), words[end].end())
    if best is None or best[0] < threshold:
        return text
    _, start_char, end_char = best
    return text[:start_char] + term_en + text[end_char:]


def clean_prediction(text: str) -> str:
    text = re.sub(r"\[\s*/?\s*[A-Za-z_]+\s*\]", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([])}])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_source_rows(root: Path, limit_train: int | None = None, limit_eval: int | None = None) -> dict[str, list[dict[str, Any]]]:
    data_dir = root / "data"
    if not data_dir.exists():
        data_dir = root / "generated_materials" / "data"
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "dev", "test"]:
        rows = read_csv(data_dir / f"{split}.csv")
        for row in rows:
            row["entities"] = parse_entities(row.get("entities"))
            row["entities_gold"] = parse_entities(row.get("entities_gold"))
            row["entities_model"] = parse_entities(row.get("entities_model"))
            row["entities_eval"] = parse_entities(row.get("entities_eval")) or row["entities"]
        if split == "train" and limit_train:
            rows = rows[:limit_train]
        if split in {"dev", "test"} and limit_eval:
            rows = rows[:limit_eval]
        rows_by_split[split] = rows
    return rows_by_split


def terms_from_entities(entities: list[dict[str, str]]) -> list[Term]:
    terms: list[Term] = []
    for ent in unique_entities(entities):
        term = Term(
            zh=ent.get("zh", "").strip(),
            en=ent.get("en", "").strip(),
            entity_type=ent.get("type", "").strip(),
            aliases=[],
        )
        if term.zh and term.en and term.entity_type:
            terms.append(term)
    return sorted(terms, key=lambda t: len(t.zh), reverse=True)


def save_config(path: Path, config: ExperimentConfig, extra: dict[str, Any] | None = None) -> None:
    ensure_dir(path.parent)
    payload = asdict(config)
    payload["timestamp"] = now()
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        seed=args.seed,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        generation_max_new_tokens=args.generation_max_new_tokens,
        num_beams=args.num_beams,
        download_retries=args.download_retries,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        data_status="real_20260508_no_leak_strict",
    )


def check_env(args: argparse.Namespace) -> None:
    root = repo_root_from_args(args)
    print(f"[check-env] root={root}")
    print(f"[check-env] conda_env={os.environ.get('CONDA_DEFAULT_ENV', '(not set)')}")
    required = ["torch", "transformers", "datasets", "peft", "accelerate", "sacrebleu", "pandas", "sentencepiece"]
    for package in required:
        mod = importlib.import_module(package)
        version = getattr(mod, "__version__", "unknown")
        print(f"[check-env] {package}={version}")
    optional = ["hf_xet", "sacremoses"]
    for package in optional:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"[check-env] optional {package}={version}")
        except ImportError:
            print(f"[check-env] optional {package}=not installed")

    import torch

    print(f"[check-env] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[check-env] gpu={torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"[check-env] gpu_memory_gb={props.total_memory / 1024 ** 3:.2f}")

    rows_by_split = load_source_rows(root)
    counts = {split: len(rows) for split, rows in rows_by_split.items()}
    print(f"[check-env] split_counts={counts}")
    print("[check-env] expected package split is strict/no-leak; counts may differ from 1000/250/250 after quarantine")

    term_path = root / "terminology" / "terminology_train.csv"
    if not term_path.exists():
        term_path = root / "generated_materials" / "terminology" / "terminology.csv"
    terms = load_terms(term_path)
    label_counts = Counter(term.entity_type for term in terms)
    print(f"[check-env] terminology={term_path}")
    print(f"[check-env] train_terms={len(terms)} entity_types={dict(label_counts)}")
    if not terms or not label_counts:
        raise RuntimeError("Terminology file is empty or malformed.")


def download_model(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "source.spm",
        "target.spm",
        "vocab.json",
        "pytorch_model.bin",
        "model.safetensors",
        "*.safetensors",
    ]
    print(f"[download-model] downloading {args.model_name}")
    path = with_retries(
        f"snapshot {args.model_name}",
        args.download_retries,
        lambda: snapshot_download(
            repo_id=args.model_name,
            allow_patterns=patterns,
            resume_download=True,
        ),
    )
    print(f"[download-model] cached at {path}")


def prepare_data(args: argparse.Namespace) -> None:
    root = repo_root_from_args(args)
    run_dir = run_dir_from_args(args, root)
    rows_by_split = load_source_rows(root, args.limit_train, args.limit_eval)
    prepared_dir = run_dir / "prepared"
    ensure_dir(prepared_dir)

    for split, rows in rows_by_split.items():
        plain_rows: list[dict[str, Any]] = []
        entity_rows: list[dict[str, Any]] = []
        for row in rows:
            model_entities = unique_entities(row.get("entities_model") or [])
            eval_entities = unique_entities(row.get("entities_eval") or row.get("entities") or [])
            gold_entities = unique_entities(row.get("entities_gold") or eval_entities)
            injected_source, detected_entities = inject_entities(row["zh"], terms_from_entities(model_entities))
            base = {
                "id": row["id"],
                "split": split,
                "source": row["zh"],
                "source_entity": injected_source,
                "target": row["en"],
                "text_type": row.get("text_type", ""),
                "entities": eval_entities,
                "entities_eval": eval_entities,
                "entities_model": detected_entities,
                "entities_gold": gold_entities,
                "source_status": row.get("source_status", ""),
            }
            plain_rows.append({**base, "source_for_model": row["zh"]})
            entity_rows.append({**base, "source_for_model": injected_source})

        write_jsonl(prepared_dir / f"{split}_plain.jsonl", plain_rows)
        write_jsonl(prepared_dir / f"{split}_entity.jsonl", entity_rows)
        print(f"[prepare] wrote {split}: plain={len(plain_rows)} entity={len(entity_rows)}")

    save_config(
        run_dir / "logs" / "run_config.json",
        build_config(args),
        extra={"prepared_dir": str(prepared_dir), "command": "prepare"},
    )


def import_transformer_stack():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    return torch, AutoTokenizer, AutoModelForSeq2SeqLM


def with_retries(label: str, attempts: int, func):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                print(f"[download] retry {attempt}/{attempts}: {label}", flush=True)
            return func()
        except Exception as exc:  # noqa: BLE001 - keep downloads resilient across hub backends.
            last_exc = exc
            message = str(exc)
            transient = any(
                token in message.lower()
                for token in ["incompleteread", "connection broken", "timeout", "temporarily", "chunkedencodingerror"]
            )
            if attempt >= attempts or not transient:
                raise
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"Failed after {attempts} attempts: {label}") from last_exc


def load_tokenizer(model_name: str, src_lang: str, tgt_lang: str, download_retries: int = 3):
    _, AutoTokenizer, _ = import_transformer_stack()
    tokenizer = with_retries(
        f"tokenizer {model_name}",
        download_retries,
        lambda: AutoTokenizer.from_pretrained(model_name),
    )
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = tgt_lang
    return tokenizer


def language_token_id(tokenizer: Any, tgt_lang: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if token_id is None or token_id == tokenizer.unk_token_id:
        mapping = getattr(tokenizer, "lang_code_to_id", {})
        token_id = mapping.get(tgt_lang)
    return token_id


def load_generation_model(args: argparse.Namespace, adapter_dir: Path | None = None):
    torch, _, AutoModelForSeq2SeqLM = import_transformer_stack()
    tokenizer = load_tokenizer(args.model_name, args.src_lang, args.tgt_lang, args.download_retries)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = with_retries(
        f"model {args.model_name}",
        args.download_retries,
        lambda: AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=dtype),
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def batch_iter(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def generate_predictions(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    source_field: str,
    adapter_dir: Path | None = None,
) -> list[str]:
    torch, tokenizer, model, device = load_generation_model(args, adapter_dir=adapter_dir)
    forced_bos_token_id = language_token_id(tokenizer, args.tgt_lang)
    predictions: list[str] = []

    with torch.no_grad():
        for batch in batch_iter(rows, args.infer_batch_size):
            inputs = [row[source_field] for row in batch]
            encoded = tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_source_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generate_kwargs = {
                "max_new_tokens": args.generation_max_new_tokens,
                "num_beams": args.num_beams,
            }
            if forced_bos_token_id is not None:
                generate_kwargs["forced_bos_token_id"] = forced_bos_token_id
            output_ids = model.generate(**encoded, **generate_kwargs)
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend([item.strip() for item in decoded])
            print(f"[generate] {len(predictions)}/{len(rows)}", flush=True)

    return predictions


def write_translation_file(
    path: Path,
    rows: list[dict[str, Any]],
    source_field: str,
    predictions: list[str],
    use_normalized_primary: bool,
) -> None:
    output_rows: list[dict[str, Any]] = []
    for row, pred in zip(rows, predictions):
        entities_eval = unique_entities(parse_entities(row.get("entities_eval")) or parse_entities(row.get("entities")))
        entities_model = unique_entities(parse_entities(row.get("entities_model")))
        entities_gold = unique_entities(parse_entities(row.get("entities_gold")) or entities_eval)
        pred_clean = clean_prediction(pred)
        pred_norm = apply_terminology_normalization(pred_clean, entities_eval)
        primary = pred_norm if use_normalized_primary else pred_clean
        output_rows.append(
            {
                "id": row["id"],
                "split": row["split"],
                "source": row["source"],
                "source_for_model": row[source_field],
                "reference": row["target"],
                "prediction_raw": pred,
                "prediction_clean": pred_clean,
                "prediction_normalized": pred_norm,
                "prediction": primary,
                "normalization_used": str(use_normalized_primary),
                "entities": entities_eval,
                "entities_eval": entities_eval,
                "entities_model": entities_model,
                "entities_gold": entities_gold,
            }
        )
    fields = [
        "id",
        "split",
        "source",
        "source_for_model",
        "reference",
        "prediction_raw",
        "prediction_clean",
        "prediction_normalized",
        "prediction",
        "normalization_used",
        "entities",
        "entities_eval",
        "entities_model",
        "entities_gold",
    ]
    write_csv(path, output_rows, fields)
    print(f"[translation] wrote {path}")


def translate_baseline(args: argparse.Namespace) -> None:
    root = repo_root_from_args(args)
    run_dir = run_dir_from_args(args, root)
    prepare_data(args)
    test_plain = read_jsonl(run_dir / "prepared" / "test_plain.jsonl")
    test_entity = read_jsonl(run_dir / "prepared" / "test_entity.jsonl")
    if args.limit_eval:
        test_plain = test_plain[: args.limit_eval]
        test_entity = test_entity[: args.limit_eval]

    translations_dir = run_dir / "translations"
    ensure_dir(translations_dir)

    b1_path = translations_dir / "B1_general_translation.csv"
    b3_path = translations_dir / "B3_entity_aware.csv"
    if args.force or not b1_path.exists():
        print("[translate-baseline] running B1_general_translation")
        preds = generate_predictions(args, test_plain, "source_for_model")
        write_translation_file(b1_path, test_plain, "source_for_model", preds, use_normalized_primary=False)
    else:
        print(f"[translate-baseline] skip existing {b1_path}")

    if args.force or not b3_path.exists():
        print("[translate-baseline] running B3_entity_aware")
        preds = generate_predictions(args, test_entity, "source_for_model")
        write_translation_file(b3_path, test_entity, "source_for_model", preds, use_normalized_primary=False)
    else:
        print(f"[translate-baseline] skip existing {b3_path}")


def tokenize_dataset(tokenizer: Any, rows: list[dict[str, Any]], args: argparse.Namespace):
    from datasets import Dataset

    dataset = Dataset.from_list(rows)

    def preprocess(batch: dict[str, list[Any]]) -> dict[str, Any]:
        inputs = batch["source_for_model"]
        targets = batch["target"]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
            padding=False,
        )
        try:
            labels = tokenizer(
                text_target=targets,
                max_length=args.max_target_length,
                truncation=True,
                padding=False,
            )
        except TypeError:
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=args.max_target_length,
                    truncation=True,
                    padding=False,
                )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    columns = dataset.column_names
    return dataset.map(preprocess, batched=True, remove_columns=columns)


def make_training_args(args: argparse.Namespace, output_dir: Path, retry_batch_size: int | None = None, retry_grad_accum: int | None = None):
    from transformers import Seq2SeqTrainingArguments

    train_bs = retry_batch_size if retry_batch_size is not None else args.train_batch_size
    grad_accum = retry_grad_accum if retry_grad_accum is not None else args.gradient_accumulation_steps
    kwargs = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        predict_with_generate=False,
        fp16=args.fp16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
    )
    try:
        return Seq2SeqTrainingArguments(eval_strategy="epoch", **kwargs)
    except TypeError:
        return Seq2SeqTrainingArguments(evaluation_strategy="epoch", **kwargs)


def train_one_lora(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    output_dir: Path,
    retry_batch_size: int | None = None,
    retry_grad_accum: int | None = None,
) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, set_seed

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model_name, args.src_lang, args.tgt_lang, args.download_retries)
    dtype = torch.float16 if torch.cuda.is_available() and args.fp16 else torch.float32
    model = with_retries(
        f"model {args.model_name}",
        args.download_retries,
        lambda: AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=dtype),
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            def make_inputs_require_grad(_module, _input, output):
                output.requires_grad_(True)

            input_embeddings.register_forward_hook(make_inputs_require_grad)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_train = tokenize_dataset(tokenizer, train_rows, args)
    tokenized_dev = tokenize_dataset(tokenizer, dev_rows, args)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    training_args = make_training_args(args, output_dir, retry_batch_size, retry_grad_accum)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    ensure_dir(output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train-lora] saved adapter to {output_dir}")


def train_lora(args: argparse.Namespace) -> None:
    root = repo_root_from_args(args)
    run_dir = run_dir_from_args(args, root)
    prepare_data(args)
    prepared = run_dir / "prepared"
    models_dir = run_dir / "models"
    ensure_dir(models_dir)

    jobs = [
        ("domain_lora", "plain"),
        ("entity_domain_lora", "entity"),
    ]
    variant = getattr(args, "variant", "both")
    if variant != "both":
        jobs = [job for job in jobs if job[0] == variant]

    for adapter_name, data_kind in jobs:
        output_dir = models_dir / adapter_name
        if output_dir.exists() and not args.force:
            print(f"[train-lora] skip existing {output_dir}")
            continue
        train_rows = read_jsonl(prepared / f"train_{data_kind}.jsonl")
        dev_rows = read_jsonl(prepared / f"dev_{data_kind}.jsonl")
        print(f"[train-lora] training {adapter_name} with {len(train_rows)} train / {len(dev_rows)} dev rows")
        try:
            train_one_lora(args, train_rows, dev_rows, output_dir)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "cuda" not in message:
                raise
            print("[train-lora] CUDA memory issue detected; retrying with batch_size=1 and grad_accum=16")
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            train_one_lora(args, train_rows, dev_rows, output_dir, retry_batch_size=1, retry_grad_accum=16)


def generate_lora(args: argparse.Namespace) -> None:
    root = repo_root_from_args(args)
    run_dir = run_dir_from_args(args, root)
    prepared = run_dir / "prepared"
    translations_dir = run_dir / "translations"
    ensure_dir(translations_dir)

    jobs = [
        ("B2_domain_adapted", "plain", run_dir / "models" / "domain_lora", False),
        ("Ours_entity_aware_domain_adapted", "entity", run_dir / "models" / "entity_domain_lora", False),
    ]
    variant = getattr(args, "variant", "both")
    if variant != "both":
        jobs = [job for job in jobs if job[0] == variant]

    for model_id, data_kind, adapter_dir, use_norm in jobs:
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")
        test_rows = read_jsonl(prepared / f"test_{data_kind}.jsonl")
        if args.limit_eval:
            test_rows = test_rows[: args.limit_eval]
        output_path = translations_dir / f"{model_id}.csv"
        if output_path.exists() and not args.force:
            print(f"[generate-lora] skip existing {output_path}")
            continue
        print(f"[generate-lora] generating {model_id}")
        preds = generate_predictions(args, test_rows, "source_for_model", adapter_dir=adapter_dir)
        write_translation_file(output_path, test_rows, "source_for_model", preds, use_normalized_primary=use_norm)


def translation_files(run_dir: Path) -> dict[str, Path]:
    return {
        "B1_general_translation": run_dir / "translations" / "B1_general_translation.csv",
        "B2_domain_adapted": run_dir / "translations" / "B2_domain_adapted.csv",
        "B3_entity_aware": run_dir / "translations" / "B3_entity_aware.csv",
        "Ours_entity_aware_domain_adapted": run_dir / "translations" / "Ours_entity_aware_domain_adapted.csv",
    }


def score_sacrebleu(preds: list[str], refs: list[str]) -> dict[str, float]:
    import sacrebleu

    bleu = sacrebleu.metrics.BLEU(effective_order=True).corpus_score(preds, [refs]).score
    chrf = sacrebleu.metrics.CHRF().corpus_score(preds, [refs]).score
    ter = sacrebleu.metrics.TER().corpus_score(preds, [refs]).score
    return {"BLEU": bleu, "chrF": chrf, "TER": ter}


def term_lookup_from_entities(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        for ent in parse_entities(row.get("entities")):
            if ent["zh"] and ent["en"]:
                lookup[ent["zh"]] = {"en": ent["en"], "type": ent.get("type", "")}
    return lookup


def score_terms(rows: list[dict[str, Any]], prediction_field: str = "prediction") -> tuple[float, float, list[dict[str, Any]]]:
    total = 0
    hit = 0
    by_type_total: Counter[str] = Counter()
    by_type_hit: Counter[str] = Counter()
    term_occurrences: dict[str, list[int]] = defaultdict(list)
    term_info: dict[str, dict[str, str]] = {}

    for row in rows:
        prediction = row[prediction_field]
        entities = unique_entities(parse_entities(row.get("entities_eval")) or parse_entities(row.get("entities")))
        for ent in entities:
            total += 1
            etype = ent.get("type", "")
            by_type_total[etype] += 1
            matched = int(contains_term(prediction, ent["en"]))
            hit += matched
            by_type_hit[etype] += matched
            term_occurrences[ent["zh"]].append(matched)
            term_info[ent["zh"]] = {"en": ent["en"], "type": etype}

    ta = hit / total * 100 if total else 0.0
    repeated_hits = 0
    repeated_total = 0
    for values in term_occurrences.values():
        if len(values) >= 2:
            repeated_hits += sum(values)
            repeated_total += len(values)
    tcr = repeated_hits / repeated_total * 100 if repeated_total else 0.0

    entity_rows: list[dict[str, Any]] = []
    for etype in sorted(by_type_total):
        entity_rows.append(
            {
                "entity_type": etype,
                "term_occurrences": by_type_total[etype],
                "hits": by_type_hit[etype],
                "TA": round(by_type_hit[etype] / by_type_total[etype] * 100, 4) if by_type_total[etype] else 0.0,
            }
        )
    return ta, tcr, entity_rows


def evaluate_one(model_id: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_csv(path)
    refs = [row["reference"] for row in rows]
    preds = [row["prediction"] for row in rows]
    scores = score_sacrebleu(preds, refs)
    ta, tcr, entity_rows = score_terms(rows, "prediction")
    raw_field = "prediction_clean" if rows and "prediction_clean" in rows[0] else "prediction_raw"
    raw_ta, raw_tcr, _ = score_terms(rows, raw_field)
    summary = {
        "model_id": model_id,
        "n": len(rows),
        "BLEU": round(scores["BLEU"], 4),
        "chrF": round(scores["chrF"], 4),
        "TER": round(scores["TER"], 4),
        "TA": round(ta, 4),
        "TCR": round(tcr, 4),
        "raw_TA": round(raw_ta, 4),
        "raw_TCR": round(raw_tcr, 4),
        "normalization_used": rows[0].get("normalization_used", "") if rows else "",
    }
    for row in entity_rows:
        row["model_id"] = model_id
    return summary, entity_rows, rows


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root_from_args(args)
    run_dir = run_dir_from_args(args, root)
    metrics_dir = run_dir / "metrics"
    paper_dir = run_dir / "paper_tables"
    ensure_dir(metrics_dir)
    ensure_dir(paper_dir)

    summaries: list[dict[str, Any]] = []
    entity_metric_rows: list[dict[str, Any]] = []
    for model_id, path in translation_files(run_dir).items():
        if not path.exists():
            print(f"[evaluate] missing {path}; skipping {model_id}")
            continue
        summary, entity_rows, rows = evaluate_one(model_id, path)
        summaries.append(summary)
        entity_metric_rows.extend(entity_rows)

    if not summaries:
        raise RuntimeError("No translation files found for evaluation.")

    result_fields = ["model_id", "n", "BLEU", "chrF", "TER", "TA", "TCR", "raw_TA", "raw_TCR", "normalization_used"]
    write_csv(metrics_dir / "metrics_summary.csv", summaries, result_fields)

    entity_fields = ["model_id", "entity_type", "term_occurrences", "hits", "TA"]
    write_csv(metrics_dir / "entity_type_metrics.csv", entity_metric_rows, entity_fields)

    ablation_rows: list[dict[str, Any]] = []
    summary_by_id = {row["model_id"]: row for row in summaries}
    ablation_map = [
        ("Full method", "None", "Ours_entity_aware_domain_adapted"),
        ("w/o entity enhancement", "Entity tags / entity-aware input", "B2_domain_adapted"),
        ("w/o domain adaptation", "LoRA domain adaptation", "B3_entity_aware"),
    ]
    for setting, removed, model_id in ablation_map:
        base = dict(summary_by_id.get(model_id, {}))
        if not base:
            continue
        ablation_rows.append(
            {
                "setting": setting,
                "removed_component": removed,
                "source_model": model_id,
                "BLEU": base.get("BLEU", ""),
                "chrF": base.get("chrF", ""),
                "TER": base.get("TER", ""),
                "TA": base.get("TA", ""),
                "TCR": base.get("TCR", ""),
            }
        )
    write_csv(metrics_dir / "ablation_summary.csv", ablation_rows, ["setting", "removed_component", "source_model", "BLEU", "chrF", "TER", "TA", "TCR"])

    md = [
        "# NLLB-LoRA TCM Translation Results",
        "",
        "> Data status: verified YYQ 2026-05-08 strict/no-leak real-data split.",
        "",
        "## Main Results",
        "",
        markdown_table(summaries, ["model_id", "BLEU", "chrF", "TER", "TA", "TCR"]),
        "",
        "## Ablation Results",
        "",
        markdown_table(ablation_rows, ["setting", "removed_component", "BLEU", "chrF", "TER", "TA", "TCR"]),
        "",
        "Note: post-hoc terminology normalization is disabled in this package, so no separate normalization ablation is reported.",
        "",
    ]
    (paper_dir / "results.md").write_text("\n".join(md), encoding="utf-8-sig")
    print(f"[evaluate] wrote {metrics_dir / 'metrics_summary.csv'}")
    print(f"[evaluate] wrote {paper_dir / 'results.md'}")

    success = check_success(summaries)
    payload = {
        "success": success,
        "summaries": summaries,
        "metrics_path": str(metrics_dir / "metrics_summary.csv"),
    }
    (metrics_dir / "evaluation_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"[evaluate] success={success}")
    return payload


def check_success(summaries: list[dict[str, Any]]) -> bool:
    by_id = {row["model_id"]: row for row in summaries}
    ours = by_id.get("Ours_entity_aware_domain_adapted")
    if not ours:
        return False
    baselines = [row for model_id, row in by_id.items() if model_id != "Ours_entity_aware_domain_adapted"]
    if not baselines:
        return False
    ta_ok = all(float(ours["TA"]) >= float(row["TA"]) for row in baselines)
    tcr_ok = all(float(ours["TCR"]) >= float(row["TCR"]) for row in baselines)
    common_metric_ok = any(
        all(float(ours[metric]) >= float(row[metric]) for row in baselines)
        for metric in ["BLEU", "chrF"]
    )
    ter_ok = all(float(ours["TER"]) <= float(row["TER"]) for row in baselines)
    return bool(ta_ok and tcr_ok and (common_metric_ok or ter_ok))


def all_pipeline(args: argparse.Namespace) -> None:
    check_env(args)
    prepare_data(args)
    translate_baseline(args)
    train_lora(args)
    generate_lora(args)
    status = evaluate(args)
    if status["success"]:
        return

    if not args.auto_tune:
        print("[all] success criteria not met; auto-tune disabled for paper-rigorous runs")
        return

    print("[all] success criteria not met; running one tuned Ours retry")
    tuned_args = argparse.Namespace(**vars(args))
    tuned_args.lora_r = 16
    tuned_args.lora_alpha = 32
    tuned_args.learning_rate = 1e-4
    tuned_args.epochs = max(float(args.epochs), 10.0)
    tuned_args.variant = "entity_domain_lora"
    tuned_args.force = True
    train_lora(tuned_args)

    gen_args = argparse.Namespace(**vars(args))
    gen_args.variant = "Ours_entity_aware_domain_adapted"
    gen_args.force = True
    generate_lora(gen_args)
    final_status = evaluate(args)
    root = repo_root_from_args(args)
    run_dir = run_dir_from_args(args, root)
    log_path = run_dir / "logs" / "run_config.json"
    config = json.loads(log_path.read_text(encoding="utf-8-sig")) if log_path.exists() else {}
    config["auto_tune"] = {
        "triggered": True,
        "lora_r": tuned_args.lora_r,
        "lora_alpha": tuned_args.lora_alpha,
        "learning_rate": tuned_args.learning_rate,
        "epochs": tuned_args.epochs,
        "success_after_tune": final_status["success"],
    }
    log_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--src-lang", default=SRC_LANG)
    parser.add_argument("--tgt-lang", default=TGT_LANG)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-source-length", type=int, default=192)
    parser.add_argument("--max-target-length", type=int, default=192)
    parser.add_argument("--generation-max-new-tokens", type=int, default=192)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--infer-batch-size", type=int, default=4)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLLB-LoRA TCM translation experiment pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["check-env", "download-model", "prepare", "translate-baseline", "evaluate", "all"]:
        sub = subparsers.add_parser(command)
        add_common_args(sub)
        if command == "all":
            sub.add_argument("--auto-tune", action="store_true", help="Optional exploratory retry. Do not use for paper main results.")

    train = subparsers.add_parser("train-lora")
    add_common_args(train)
    train.add_argument("--variant", choices=["both", "domain_lora", "entity_domain_lora"], default="both")

    generate = subparsers.add_parser("generate-lora")
    add_common_args(generate)
    generate.add_argument(
        "--variant",
        choices=["both", "B2_domain_adapted", "Ours_entity_aware_domain_adapted"],
        default="both",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "check-env":
            check_env(args)
        elif args.command == "download-model":
            download_model(args)
        elif args.command == "prepare":
            prepare_data(args)
        elif args.command == "translate-baseline":
            translate_baseline(args)
        elif args.command == "train-lora":
            train_lora(args)
        elif args.command == "generate-lora":
            generate_lora(args)
        elif args.command == "evaluate":
            evaluate(args)
        elif args.command == "all":
            all_pipeline(args)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
