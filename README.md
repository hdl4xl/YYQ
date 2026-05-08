# YYQEI TCM Translation Experiment

This repository's current paper package is **V2**. It contains the strict/no-leak
real-data experiments for Traditional Chinese Medicine Chinese-to-English
translation.

## Current Version: V2

- Package: [`V2/`](V2/)
- Data root: [`V2/YYQEI_20260508_NLLB_ready/`](V2/YYQEI_20260508_NLLB_ready/)
- Strict split: train 990 / dev 249 / test 250
- Test set size: 250
- Backbones: `facebook/nllb-200-distilled-600M` and `Helsinki-NLP/opus-mt-zh-en`
- No-leak policy: train-only terminology for model input; audit-only full terminology
- Post-hoc `Terminology: ...` normalization is disabled
- Model checkpoints and the original PDF are excluded from GitHub

## Results

All results below are from the same V2 strict test set. `normalization_used=False`
for all settings.

### NLLB-200-distilled-600M

| Setting | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1_general_translation | 4.8841 | 25.9489 | 95.8452 | 12.3529 | 16.5025 |
| B2_domain_adapted | 16.9691 | 43.8895 | 68.8210 | 31.6176 | 44.0887 |
| B3_entity_aware | 18.3225 | 43.3200 | 75.8168 | 56.6176 | 71.9212 |
| Ours_entity_aware_domain_adapted | 31.4978 | 52.5205 | 55.6463 | 60.7353 | 77.0936 |

### Helsinki-NLP/opus-mt-zh-en

| Setting | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1_general_translation | 6.7578 | 27.2359 | 83.5227 | 15.8824 | 21.6749 |
| B2_domain_adapted | 13.0818 | 39.5669 | 76.1009 | 30.0000 | 43.3498 |
| B3_entity_aware | 10.2387 | 36.9979 | 73.4730 | 44.2647 | 59.6059 |
| Ours_entity_aware_domain_adapted | 20.1525 | 44.6517 | 64.4886 | 48.2353 | 64.7783 |

## Quick Start

```powershell
cd V2
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_01_validate.py
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_02_check_env.py
```

Run the full experiments:

```powershell
cd V2
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_10_run_nllb.py
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_11_run_helsinki.py
```

## Versions

- [`V2/`](V2/): current recommended package for the paper experiment.
- [`V1/`](V1/): historical archive of the earlier package.

For new experiments, use `V2/`.
