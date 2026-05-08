# NLLB-LoRA TCM Translation Results

> Data status: verified YYQ 2026-05-08 strict/no-leak real-data split.

## Main Results

| model_id | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- |
| B1_general_translation | 4.8841 | 25.9489 | 95.8452 | 12.3529 | 16.5025 |
| B2_domain_adapted | 17.1078 | 44.4764 | 67.9688 | 30.5882 | 42.8571 |
| B3_entity_aware | 18.3225 | 43.32 | 75.8168 | 56.6176 | 71.9212 |
| Ours_entity_aware_domain_adapted | 31.3333 | 52.6118 | 55.1847 | 60.4412 | 76.601 |

## Ablation Results

| setting | removed_component | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- | --- |
| Full method | None | 31.3333 | 52.6118 | 55.1847 | 60.4412 | 76.601 |
| w/o entity enhancement | Entity tags / entity-aware input | 17.1078 | 44.4764 | 67.9688 | 30.5882 | 42.8571 |
| w/o domain adaptation | LoRA domain adaptation | 18.3225 | 43.32 | 75.8168 | 56.6176 | 71.9212 |

Note: post-hoc terminology normalization is disabled in this package, so no separate normalization ablation is reported.
