# YYQEI TCM Translation Experiment

This repository contains the reproducible scripts, cleaned real-data corpus, and experiment outputs for the YYQEI Traditional Chinese Medicine Chinese-to-English translation study.

## Current Data

- Final data root: `YYQEI_NLLB_ready/`
- Corpus size: 1,397 sentence pairs
- Split: train 906 / dev 242 / test 249
- Terminology entries: 315
- BIO annotations: full `BIO_Spans` and `BIO_Char_Full`

The original source PDF and model checkpoints are intentionally excluded from GitHub because of size and copyright/storage constraints.

## Main Scripts

- `scripts/build_nllb_ready_data.py`: builds the NLLB-ready real-data package from cleaned materials.
- `scripts/nllb_tcm_experiment.py`: prepares data, runs baseline translation, LoRA training, generation, and evaluation.

## Final Experiment Outputs

Real-data outputs are kept under:

- `runs/helsinki_real_data/`
- `runs/nllb_real_data_full/`

Included outputs:

- prepared JSONL files
- test translations
- metrics
- paper-ready result tables

Excluded outputs:

- model adapters and checkpoints
- optimizer states
- tokenizer/model cache files

## Best Current Result

The strongest result is from `facebook/nllb-200-distilled-600M` with entity-aware domain adaptation:

| Model | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| NLLB baseline | 6.4683 | 30.6957 | 90.9983 | 28.1826 | 29.0741 |
| Domain LoRA | 7.9401 | 33.2285 | 98.3960 | 38.4548 | 40.0000 |
| Entity-aware baseline | 8.6157 | 36.1553 | 88.2882 | 92.4495 | 92.4074 |
| Full method | 11.6486 | 40.1288 | 97.7185 | 99.7366 | 99.8148 |

## Environment

Experiments were run with the `YYQEI` Conda environment:

```powershell
D:\Anaconda\Scripts\conda.exe run -n YYQEI python scripts\nllb_tcm_experiment.py check-env --root D:\Desktop\YYQEI\YYQEI_NLLB_ready
```

