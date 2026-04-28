# NLLB-ready YYQEI Data

- total_rows: 1397
- split_counts: {'train': 906, 'dev': 242, 'test': 249}
- duplicate_zh_norm_rows: 0
- duplicate_en_norm_rows: 0
- known_bad_en_rows: []
- ascii_period_zh_rows: []
- term_coverage_missing: []
- bio_span_errors: 0
- bio_char_reconstruction_errors: 0
- corrections_applied: 18

Use with the existing script:

```powershell
python scripts\nllb_tcm_experiment.py prepare --root D:\Desktop\YYQEI\YYQEI_NLLB_ready --run-dir runs\nllb_real_data
```