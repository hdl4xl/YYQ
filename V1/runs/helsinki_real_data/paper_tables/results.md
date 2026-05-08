# NLLB-LoRA TCM Translation Results

> Data status: synthetic placeholder for pipeline development; replace with verified YYQ data before submission.

## Main Results

| model_id | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- |
| B1_general_translation | 6.123 | 30.0849 | 89.6156 | 31.0799 | 32.2222 |
| B2_domain_adapted | 7.2617 | 31.7402 | 102.3783 | 40.5619 | 42.4074 |
| B3_entity_aware | 4.2172 | 28.8132 | 88.7306 | 65.1449 | 66.2037 |
| Ours_entity_aware_domain_adapted | 6.0999 | 37.0037 | 114.2284 | 99.2976 | 99.2593 |

## Ablation Results

| setting | removed_component | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- | --- |
| Full method | None | 6.0999 | 37.0037 | 114.2284 | 99.2976 | 99.2593 |
| w/o entity enhancement | Entity tags / entity-aware input | 7.2617 | 31.7402 | 102.3783 | 40.5619 | 42.4074 |
| w/o domain adaptation | LoRA domain adaptation | 4.2172 | 28.8132 | 88.7306 | 65.1449 | 66.2037 |
| w/o terminology normalization | Terminology normalization post-processing | 6.0453 | 31.4384 | 106.3606 | 43.8982 | 45.3704 |
