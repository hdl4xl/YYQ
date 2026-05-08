# NLLB-LoRA TCM Translation Results

> Data status: synthetic placeholder for pipeline development; replace with verified YYQ data before submission.

## Main Results

| model_id | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- |
| B1_general_translation | 6.4683 | 30.6957 | 90.9983 | 28.1826 | 29.0741 |
| B2_domain_adapted | 7.9401 | 33.2285 | 98.396 | 38.4548 | 40.0 |
| B3_entity_aware | 8.6157 | 36.1553 | 88.2882 | 92.4495 | 92.4074 |
| Ours_entity_aware_domain_adapted | 11.6486 | 40.1288 | 97.7185 | 99.7366 | 99.8148 |

## Ablation Results

| setting | removed_component | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- | --- |
| Full method | None | 11.6486 | 40.1288 | 97.7185 | 99.7366 | 99.8148 |
| w/o entity enhancement | Entity tags / entity-aware input | 7.9401 | 33.2285 | 98.396 | 38.4548 | 40.0 |
| w/o domain adaptation | LoRA domain adaptation | 8.6157 | 36.1553 | 88.2882 | 92.4495 | 92.4074 |
| w/o terminology normalization | Terminology normalization post-processing | 12.0557 | 38.7463 | 96.1007 | 81.3872 | 82.1296 |
