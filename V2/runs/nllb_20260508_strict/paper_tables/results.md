# NLLB-LoRA TCM Translation Results

> Data status: verified YYQ 2026-05-08 strict/no-leak real-data split.

## Main Results

| model_id | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- |
| B1_general_translation | 4.8841 | 25.9489 | 95.8452 | 12.3529 | 16.5025 |
| B2_domain_adapted | 16.9691 | 43.8895 | 68.821 | 31.6176 | 44.0887 |
| B3_entity_aware | 18.3225 | 43.32 | 75.8168 | 56.6176 | 71.9212 |
| Ours_entity_aware_domain_adapted | 31.4978 | 52.5205 | 55.6463 | 60.7353 | 77.0936 |

## Ablation Results

| setting | removed_component | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- | --- |
| Full method | None | 31.4978 | 52.5205 | 55.6463 | 60.7353 | 77.0936 |
| w/o entity enhancement | Entity tags / entity-aware input | 16.9691 | 43.8895 | 68.821 | 31.6176 | 44.0887 |
| w/o domain adaptation | LoRA domain adaptation | 18.3225 | 43.32 | 75.8168 | 56.6176 | 71.9212 |

Note: post-hoc terminology normalization is disabled in this package, so no separate normalization ablation is reported.
