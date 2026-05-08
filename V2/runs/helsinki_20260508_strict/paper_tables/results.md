# NLLB-LoRA TCM Translation Results

> Data status: verified YYQ 2026-05-08 strict/no-leak real-data split.

## Main Results

| model_id | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- |
| B1_general_translation | 6.7578 | 27.2359 | 83.5227 | 15.8824 | 21.6749 |
| B2_domain_adapted | 13.0818 | 39.5669 | 76.1009 | 30.0 | 43.3498 |
| B3_entity_aware | 10.2387 | 36.9979 | 73.473 | 44.2647 | 59.6059 |
| Ours_entity_aware_domain_adapted | 20.1525 | 44.6517 | 64.4886 | 48.2353 | 64.7783 |

## Ablation Results

| setting | removed_component | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- | --- |
| Full method | None | 20.1525 | 44.6517 | 64.4886 | 48.2353 | 64.7783 |
| w/o entity enhancement | Entity tags / entity-aware input | 13.0818 | 39.5669 | 76.1009 | 30.0 | 43.3498 |
| w/o domain adaptation | LoRA domain adaptation | 10.2387 | 36.9979 | 73.473 | 44.2647 | 59.6059 |

Note: post-hoc terminology normalization is disabled in this package, so no separate normalization ablation is reported.
