# V3 Multi-Seed Stability

- Seeds: 42, 13, 2026
- Values are mean ± sample standard deviation across seeds.

| model_id | n_seeds | BLEU | chrF | TER | TA | TCR |
| --- | --- | --- | --- | --- | --- | --- |
| B1_general_translation | 3 | 4.8841 ± 0.0000 | 25.9489 ± 0.0000 | 95.8452 ± 0.0000 | 12.3529 ± 0.0000 | 16.5025 ± 0.0000 |
| B2_domain_adapted | 3 | 17.4398 ± 0.4606 | 44.5986 ± 0.1064 | 68.3239 ± 0.3095 | 31.7647 ± 1.0189 | 44.5813 ± 1.4982 |
| B3_entity_aware | 3 | 18.3225 ± 0.0000 | 43.3200 ± 0.0000 | 75.8168 ± 0.0000 | 56.6176 ± 0.0000 | 71.9212 ± 0.0000 |
| Ours_entity_aware_domain_adapted | 3 | 31.2881 ± 0.0502 | 52.6101 ± 0.1331 | 55.4214 ± 0.3503 | 60.7843 ± 0.3061 | 77.1757 ± 0.6198 |
