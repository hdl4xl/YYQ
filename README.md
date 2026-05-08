# YYQEI TCM Translation Experiment

This repository keeps versioned experiment packages for the YYQEI Traditional
Chinese Medicine Chinese-to-English translation study.

## Current Recommended Version

Use **V2** for the current paper experiment:

- Package: [`V2/`](V2/)
- Data root: [`V2/YYQEI_20260508_NLLB_ready/`](V2/YYQEI_20260508_NLLB_ready/)
- Strict split: train 990 / dev 249 / test 250
- Backbones: `facebook/nllb-200-distilled-600M` and `Helsinki-NLP/opus-mt-zh-en`
- No-leak policy: train-only terminology for model input; audit-only full terminology
- Post-hoc `Terminology: ...` normalization is disabled
- Model checkpoints and original PDF are excluded from GitHub

Best current full-method result on the strict test set:

| Backbone | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| NLLB-200-distilled-600M | 31.4978 | 52.5205 | 55.6463 | 60.7353 | 77.0936 |
| Helsinki-NLP/opus-mt-zh-en | 20.1525 | 44.6517 | 64.4886 | 48.2353 | 64.7783 |

Start here:

```powershell
cd V2
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_01_validate.py
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_02_check_env.py
```

## Version History

| Version | Location | Status | Notes |
| --- | --- | --- | --- |
| V2 | [`V2/`](V2/) | Current recommended | Strict/no-leak 2026-05-08 package with NLLB and Helsinki comparison |
| V1 | repository root legacy files | Historical | Earlier 1,397-sample package: train 906 / dev 242 / test 249 |

The `V2/` directory is intentionally kept as a versioned package so older files
remain available instead of being overwritten. Future stable versions can be
added as `V3/`, `V4/`, and so on.

## Repository Layout

- [`V2/`](V2/): current clean reproducible package.
- `YYQEI_NLLB_ready/`: legacy V1 NLLB-ready data.
- `runs/`: legacy V1 experiment outputs.
- `scripts/`: legacy V1 scripts.
- `tools/`: legacy utilities.

For new experiments or paper results, use `V2/` unless a document explicitly
states otherwise.
