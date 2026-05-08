# YYQEI 2026-05-08 Clean Experiment Package

This directory is a GitHub-ready package for reproducing the strict/no-leak
Traditional Chinese Medicine zh-en translation experiments.

The package is built from the verified 2026-05-08 real-data split and keeps the
experiment inputs, scripts, metrics, paper tables, and model translations. It
does not include trained LoRA adapters, checkpoints, optimizer states, tokenizer
caches, or the original PDF source.

## What Is Included

- `YYQEI_20260508_NLLB_ready/`: strict NLLB-ready real-data package.
- `scripts/`: validation and experiment scripts.
- `runs/nllb_20260508_strict/`: NLLB metrics, prepared files, and translations.
- `runs/helsinki_20260508_strict/`: Helsinki metrics, prepared files, and translations.
- `docs/`: experiment report and clean-package notes.
- `requirements.txt` and `environment.yml`: reproducible environment hints.
- `LICENSE` and `DATA_LICENSE.md`: code and data-use notices.

## Data Split

The main experiments use `YYQEI_20260508_NLLB_ready/data/`, which is the strict
no-leak split:

| split | rows |
| --- | ---: |
| train | 990 |
| dev | 249 |
| test | 250 |

The original `1000/250/250` files are retained under `data_full/` for audit only.
The strict split removes cross-split exact or near-duplicate samples.

## No-Leak Policy

- Test-time entity prompting uses `entities_model` only.
- `entities_model` is constrained by `terminology/terminology_train.csv`.
- `terminology_all.csv` is retained for audit/gold metadata only.
- TA/TCR is evaluated with `entities_eval`, not test-time glossary injection.
- Post-hoc `Terminology: ...` suffix injection is disabled.
- `all` does not use test-driven auto-tune unless `--auto-tune` is explicitly supplied.
- Paper main results should not use `--auto-tune`.

## Quick Validation

Run these commands from this directory:

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_01_validate.py
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_02_check_env.py
```

If the `YYQEI` environment is already activated, the shorter form also works:

```powershell
python .\run_01_validate.py
python .\run_02_check_env.py
```

The expected strict counts are:

```text
train=990, dev=249, test=250
```

## Run Experiments

NLLB-200 distilled 600M:

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_10_run_nllb.py
```

Helsinki OPUS-MT:

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_11_run_helsinki.py
```

The default runtime is:

```text
D:\Anaconda\Scripts\conda.exe
conda env: YYQEI
```

## Existing Results

Main full-method results on the same strict test set:

| backbone | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| NLLB-200-distilled-600M | 31.4978 | 52.5205 | 55.6463 | 60.7353 | 77.0936 |
| Helsinki-NLP/opus-mt-zh-en | 20.1525 | 44.6517 | 64.4886 | 48.2353 | 64.7783 |

NLLB is stronger on this dataset, while both backbones show clear gains from
domain adaptation plus entity-aware input.

## GitHub Package Boundary

This package intentionally excludes model artifacts. Users should download
`facebook/nllb-200-distilled-600M` and `Helsinki-NLP/opus-mt-zh-en` through
Hugging Face and comply with the corresponding model cards and licenses.

See `docs/GITHUB_CLEAN_PACKAGE.md` for the exact exclusion policy.
