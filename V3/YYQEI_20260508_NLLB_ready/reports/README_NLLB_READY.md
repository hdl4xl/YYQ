# YYQEI 2026-05-08 NLLB-ready Data

This folder was generated from `source_materials` for no-leak TCM Chinese-to-English MT experiments.

## Main Policy
- `data/` is the strict default split used by the experiment script.
- `data_full/` preserves the original 1000/250/250 split.
- `data_strict/` removes cross-split exact or high-similarity near duplicates.
- Model entity tags use only `terminology/terminology_train.csv` through `entities_model`.
- TA/TCR scoring uses `entities_eval`, which requires exact Chinese source and English reference evidence.
- `terminology_all.csv` is audit/gold metadata only and must not be used for test-time prompting.
- The experiment script disables Terminology-suffix post-processing and reports raw outputs as primary results.

## Counts
- data_full: {'train': 1000, 'dev': 250, 'test': 250}
- data_strict/default data: {'train': 990, 'dev': 249, 'test': 250}
- strict removed IDs: 11
- terminology_train rows: 1285
- terminology_all rows: 1828

## Run
From the package root:

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python scripts\validate_ready_package.py --root .\YYQEI_20260508_NLLB_ready
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python scripts\nllb_tcm_experiment.py check-env --root .\YYQEI_20260508_NLLB_ready --run-dir .\runs\nllb_20260508
```
