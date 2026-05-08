# Package Validation Summary

Validation date: 2026-05-08

Package root: `D:\Desktop\YYQEI\V2`

## Data Status

- Ready data: `YYQEI_20260508_NLLB_ready/`
- Default experiment split: `data/` strict split
- Strict counts: `train=990, dev=249, test=250`
- Full audit counts in `data_full/`: `train=1000, dev=250, test=250`
- Strict split removes 11 cross-split exact or near-duplicate samples.

## No-Leak Status

- Model input uses `entities_model` only.
- `entities_model` is constrained by `terminology/terminology_train.csv`.
- `terminology_train.csv` contains no `TCM_DEV_` or `TCM_TEST_` evidence IDs.
- `terminology_all.csv` is audit/gold metadata only.
- TA/TCR uses `entities_eval`, not test-time glossary injection.
- Post-hoc `Terminology: ...` suffix injection is disabled.
- `normalization_used=False` for all main translation outputs.
- `all` does not auto-tune unless `--auto-tune` is explicitly supplied.

## Verified Experiment Outputs

Both runs contain:

- `prepared/*.jsonl`
- `translations/*.csv`
- `metrics/metrics_summary.csv`
- `metrics/ablation_summary.csv`
- `metrics/entity_type_metrics.csv`
- `metrics/evaluation_status.json`
- `paper_tables/results.md`
- `logs/run_config.json`

Both runs have four test translation files with 250 rows each.

## Verified Commands

These commands passed on the packaging machine:

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_01_validate.py
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_02_check_env.py
```

The packaging shell did not expose bare `python` on PATH, so the verified
commands use the fixed `YYQEI` conda environment explicitly.

Key environment versions:

| package | version |
| --- | --- |
| torch | 2.8.0+cu128 |
| transformers | 4.57.6 |
| datasets | 4.5.0 |
| peft | 0.17.1 |
| accelerate | 1.10.1 |
| sacrebleu | 2.6.0 |
| pandas | 2.3.3 |
| sentencepiece | 0.2.1 |
| sacremoses | 0.1.1 |

## Main Results

| backbone | model | BLEU | chrF | TER | TA | TCR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NLLB | Ours_entity_aware_domain_adapted | 31.4978 | 52.5205 | 55.6463 | 60.7353 | 77.0936 |
| Helsinki | Ours_entity_aware_domain_adapted | 20.1525 | 44.6517 | 64.4886 | 48.2353 | 64.7783 |

NLLB outperforms Helsinki by +11.3453 BLEU, +7.8688 chrF, -8.8423 TER,
+12.5000 TA, and +12.3153 TCR on the full method.

## GitHub Clean Status

Excluded artifacts:

- `runs/**/models/`
- `runs/**/checkpoint-*`
- `*.safetensors`
- `*.pt`
- `*.pth`
- `*.bin`
- tokenizer/model cache files
- `__pycache__/`

The original PDF source and trained model weights are not included.
