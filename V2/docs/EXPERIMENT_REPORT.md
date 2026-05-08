# YYQEI 2026-05-08 Experiment Report

## Dataset

The main experiment uses the strict/no-leak split in
`YYQEI_20260508_NLLB_ready/data/`.

| split | rows |
| --- | ---: |
| train | 990 |
| dev | 249 |
| test | 250 |

The full `1000/250/250` split is retained under `data_full/` for audit only.
The strict split removes cross-split exact and near-duplicate samples.

## Experimental Setting

Two backbones are evaluated on the same real data:

- `facebook/nllb-200-distilled-600M`
- `Helsinki-NLP/opus-mt-zh-en`

Each backbone has four settings:

- `B1_general_translation`
- `B2_domain_adapted`
- `B3_entity_aware`
- `Ours_entity_aware_domain_adapted`

Post-hoc terminology normalization is disabled. No output should contain a
`Terminology:` suffix. The paper main results should not use `--auto-tune`.

## Backbone Comparison

| backbone | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| NLLB-200-distilled-600M | 31.4978 | 52.5205 | 55.6463 | 60.7353 | 77.0936 |
| Helsinki-NLP/opus-mt-zh-en | 20.1525 | 44.6517 | 64.4886 | 48.2353 | 64.7783 |

NLLB performs better than Helsinki on the strict test set:

- BLEU: +11.3453
- chrF: +7.8688
- TER: -8.8423
- TA: +12.5000
- TCR: +12.3153

## Method Effectiveness

Within NLLB, the full method improves over B1 by +26.6137 BLEU, +26.5716 chrF,
-40.1989 TER, +48.3824 TA, and +60.5911 TCR.

Within Helsinki, the full method improves over B1 by +13.3947 BLEU, +17.4158
chrF, -19.0341 TER, +32.3529 TA, and +43.1034 TCR.

This supports the core claim that domain adaptation and entity-aware input help
TCM terminology translation on the verified real-data split. NLLB is the
stronger backbone for the paper's main result.

## Reporting Caution

Do not report `w/o terminology normalization` as an ablation in this version.
Normalization is disabled by design, so such a row would duplicate the full
method and mislead readers.
