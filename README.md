# YYQEI 中医术语翻译实验

本仓库保存 YYQEI 中医基础理论中英翻译实验的多个版本。当前推荐用于投稿论文定量结果的是 **V3**；V2 保留为上一版完整实验包，V1 保留为历史归档。

## V3：当前推荐版本

V3 在 V2 strict/no-leak 数据和实验流程基础上增加了统计验证：

- 版本目录：[`V3/`](V3/)
- 数据目录：[`V3/YYQEI_20260508_NLLB_ready/`](V3/YYQEI_20260508_NLLB_ready/)
- 数据划分：train 990 / dev 249 / test 250
- 主模型：`facebook/nllb-200-distilled-600M`
- 多 seed：`42, 13, 2026`
- bootstrap：paired bootstrap resampling，5000 次
- 比较对象：Ours vs B2、Ours vs B3
- 所有输出均满足 `normalization_used=False`
- 已禁用后处理式 `Terminology: ...` 术语补词
- GitHub 不上传模型 checkpoint、LoRA 权重、optimizer 状态和 tokenizer 缓存

### V3 多 Seed 稳定性结果

数值为 `mean ± sample std`。B1 和 B3 是 deterministic inference-only baselines，因此不同 seed 下输出不变，标准差为 0；B2 和 Ours 涉及 LoRA 训练，更能体现 seed 稳定性。

| model_id | n_seeds | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1_general_translation | 3 | 4.8841 ± 0.0000 | 25.9489 ± 0.0000 | 95.8452 ± 0.0000 | 12.3529 ± 0.0000 | 16.5025 ± 0.0000 |
| B2_domain_adapted | 3 | 17.4398 ± 0.4606 | 44.5986 ± 0.1064 | 68.3239 ± 0.3095 | 31.7647 ± 1.0189 | 44.5813 ± 1.4982 |
| B3_entity_aware | 3 | 18.3225 ± 0.0000 | 43.3200 ± 0.0000 | 75.8168 ± 0.0000 | 56.6176 ± 0.0000 | 71.9212 ± 0.0000 |
| Ours_entity_aware_domain_adapted | 3 | 31.2881 ± 0.0502 | 52.6101 ± 0.1331 | 55.4214 ± 0.3503 | 60.7843 ± 0.3061 | 77.1757 ± 0.6198 |

### V3 每个 Seed 的完整指标

| seed | model_id | n | BLEU | chrF | TER | TA | TCR |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | B1_general_translation | 250 | 4.8841 | 25.9489 | 95.8452 | 12.3529 | 16.5025 |
| 42 | B2_domain_adapted | 250 | 17.1078 | 44.4764 | 67.9688 | 30.5882 | 42.8571 |
| 42 | B3_entity_aware | 250 | 18.3225 | 43.3200 | 75.8168 | 56.6176 | 71.9212 |
| 42 | Ours_entity_aware_domain_adapted | 250 | 31.3333 | 52.6118 | 55.1847 | 60.4412 | 76.6010 |
| 13 | B1_general_translation | 250 | 4.8841 | 25.9489 | 95.8452 | 12.3529 | 16.5025 |
| 13 | B2_domain_adapted | 250 | 17.9656 | 44.6488 | 68.5369 | 32.3529 | 45.3202 |
| 13 | B3_entity_aware | 250 | 18.3225 | 43.3200 | 75.8168 | 56.6176 | 71.9212 |
| 13 | Ours_entity_aware_domain_adapted | 250 | 31.2970 | 52.4762 | 55.8239 | 60.8824 | 77.0936 |
| 2026 | B1_general_translation | 250 | 4.8841 | 25.9489 | 95.8452 | 12.3529 | 16.5025 |
| 2026 | B2_domain_adapted | 250 | 17.2460 | 44.6706 | 68.4659 | 32.3529 | 45.5665 |
| 2026 | B3_entity_aware | 250 | 18.3225 | 43.3200 | 75.8168 | 56.6176 | 71.9212 |
| 2026 | Ours_entity_aware_domain_adapted | 250 | 31.2341 | 52.7424 | 55.2557 | 61.0294 | 77.8325 |

### V3 Bootstrap Significance 完整结果

正向 delta 表示 Ours 更好。BLEU、chrF、TA、TCR 使用 `Ours - baseline`；TER 使用 `baseline TER - Ours TER`。

| seed | comparison | metric | observed_delta | 95% CI | p_value | significant_p05 |
| ---: | --- | --- | ---: | --- | ---: | --- |
| 42 | Ours_vs_B2 | BLEU | 14.2256 | [10.9881, 17.6234] | 0.0002 | True |
| 42 | Ours_vs_B2 | chrF | 8.1354 | [5.8140, 10.4897] | 0.0002 | True |
| 42 | Ours_vs_B2 | TER | 12.7841 | [9.8412, 15.9309] | 0.0002 | True |
| 42 | Ours_vs_B2 | TA | 29.8529 | [25.4927, 34.2593] | 0.0002 | True |
| 42 | Ours_vs_B2 | TCR | 33.7438 | [25.0909, 36.2458] | 0.0002 | True |
| 42 | Ours_vs_B3 | BLEU | 13.0108 | [10.7599, 15.7096] | 0.0002 | True |
| 42 | Ours_vs_B3 | chrF | 9.2918 | [7.6105, 11.0481] | 0.0002 | True |
| 42 | Ours_vs_B3 | TER | 20.6321 | [15.1971, 27.2598] | 0.0002 | True |
| 42 | Ours_vs_B3 | TA | 3.8235 | [1.6081, 6.2885] | 0.0006 | True |
| 42 | Ours_vs_B3 | TCR | 4.6798 | [1.2431, 7.2861] | 0.0012 | True |
| 13 | Ours_vs_B2 | BLEU | 13.3313 | [10.1431, 16.6218] | 0.0002 | True |
| 13 | Ours_vs_B2 | chrF | 7.8275 | [5.5986, 10.0782] | 0.0002 | True |
| 13 | Ours_vs_B2 | TER | 12.7131 | [9.6277, 15.8047] | 0.0002 | True |
| 13 | Ours_vs_B2 | TA | 28.5294 | [24.3045, 32.8890] | 0.0002 | True |
| 13 | Ours_vs_B2 | TCR | 31.7734 | [23.8011, 34.7979] | 0.0002 | True |
| 13 | Ours_vs_B3 | BLEU | 12.9745 | [10.6998, 15.6563] | 0.0002 | True |
| 13 | Ours_vs_B3 | chrF | 9.1562 | [7.4447, 11.0180] | 0.0002 | True |
| 13 | Ours_vs_B3 | TER | 19.9929 | [14.4952, 26.5113] | 0.0002 | True |
| 13 | Ours_vs_B3 | TA | 4.2647 | [1.9345, 6.7724] | 0.0002 | True |
| 13 | Ours_vs_B3 | TCR | 5.1724 | [1.5570, 7.6786] | 0.0008 | True |
| 2026 | Ours_vs_B2 | BLEU | 13.9881 | [10.5744, 17.5415] | 0.0002 | True |
| 2026 | Ours_vs_B2 | chrF | 8.0718 | [5.7469, 10.3818] | 0.0002 | True |
| 2026 | Ours_vs_B2 | TER | 13.2102 | [10.0859, 16.3479] | 0.0002 | True |
| 2026 | Ours_vs_B2 | TA | 28.6765 | [24.3902, 33.0371] | 0.0002 | True |
| 2026 | Ours_vs_B2 | TCR | 32.2660 | [23.7927, 35.0263] | 0.0002 | True |
| 2026 | Ours_vs_B3 | BLEU | 12.9116 | [10.5605, 15.5683] | 0.0002 | True |
| 2026 | Ours_vs_B3 | chrF | 9.4224 | [7.7206, 11.1852] | 0.0002 | True |
| 2026 | Ours_vs_B3 | TER | 20.5611 | [15.0454, 27.2045] | 0.0002 | True |
| 2026 | Ours_vs_B3 | TA | 4.4118 | [2.1211, 6.9802] | 0.0002 | True |
| 2026 | Ours_vs_B3 | TCR | 5.9113 | [1.8483, 8.0442] | 0.0010 | True |

## V2：上一版完整实验包

V2 是当前数据与流程的基础版本，保留了同一 strict/no-leak 数据上的 NLLB 与 Helsinki 对比实验。

- 版本目录：[`V2/`](V2/)
- 数据目录：[`V2/YYQEI_20260508_NLLB_ready/`](V2/YYQEI_20260508_NLLB_ready/)
- 数据划分：train 990 / dev 249 / test 250
- 对比模型：`facebook/nllb-200-distilled-600M` 与 `Helsinki-NLP/opus-mt-zh-en`

### V2 NLLB-200-distilled-600M

| 实验设置 | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1_general_translation | 4.8841 | 25.9489 | 95.8452 | 12.3529 | 16.5025 |
| B2_domain_adapted | 16.9691 | 43.8895 | 68.8210 | 31.6176 | 44.0887 |
| B3_entity_aware | 18.3225 | 43.3200 | 75.8168 | 56.6176 | 71.9212 |
| Ours_entity_aware_domain_adapted | 31.4978 | 52.5205 | 55.6463 | 60.7353 | 77.0936 |

### V2 Helsinki-NLP/opus-mt-zh-en

| 实验设置 | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1_general_translation | 6.7578 | 27.2359 | 83.5227 | 15.8824 | 21.6749 |
| B2_domain_adapted | 13.0818 | 39.5669 | 76.1009 | 30.0000 | 43.3498 |
| B3_entity_aware | 10.2387 | 36.9979 | 73.4730 | 44.2647 | 59.6059 |
| Ours_entity_aware_domain_adapted | 20.1525 | 44.6517 | 64.4886 | 48.2353 | 64.7783 |

## V1：历史版本归档

V1 保存早期真实数据实验包，仅用于追溯历史进展，不建议作为当前投稿论文主结果。

- 版本目录：[`V1/`](V1/)
- 早期数据规模：1397 条
- 早期划分：train 906 / dev 242 / test 249

## 快速验证 V3

```powershell
cd V3
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_01_validate.py
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_02_check_env.py
```

生成 V3 统计结果：

```powershell
cd V3
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\scripts\bootstrap_significance.py --seeds 42 13 2026
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\scripts\summarize_seed_stability.py --seeds 42 13 2026
```
