# YYQEI V3：Bootstrap Significance 与多 Seed 稳定性

本目录在 V2 strict/no-leak 实验包基础上继续扩展，只用于 V3 统计增强实验。

V3 目标：

- 对 NLLB Ours 相比 B2/B3 做 paired bootstrap significance。
- 对 NLLB 做多 seed 稳定性实验。
- 不做人工作误差分析和案例分析；这两项留到 V4。

V3 不修改 V2，不覆盖 V2 输出。模型 checkpoint、LoRA adapter、optimizer 状态和 tokenizer 缓存仍不适合上传 GitHub。

## 数据划分不变

主实验使用 `YYQEI_20260508_NLLB_ready/data/`，即 strict/no-leak 数据划分：

| split | 条数 |
| --- | ---: |
| train | 990 |
| dev | 249 |
| test | 250 |

原始 `1000/250/250` 划分保留在 `data_full/` 中，仅用于审计和对照。strict split 已隔离跨 split 的 exact duplicate 和 near duplicate 样本。

## 防泄漏策略

- test-time entity prompting 只使用 `entities_model`。
- `entities_model` 受 `terminology/terminology_train.csv` 约束。
- `terminology_all.csv` 只保留为审计和 gold metadata，不进入 test-time prompting。
- TA/TCR 使用 `entities_eval` 评价，不使用 test-time glossary injection。
- 已禁用后处理式 `Terminology: ...` 术语补词。
- `all` 默认不做 test-driven auto-tune；如需探索性调参，必须显式传入 `--auto-tune`。
- 论文主结果不应使用 `--auto-tune`。

## 快速验证

在本目录运行：

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_01_validate.py
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_02_check_env.py
```

如果已经激活 `YYQEI` 环境，也可以使用简写：

```powershell
python .\run_01_validate.py
python .\run_02_check_env.py
```

预期 strict 数据计数：

```text
train=990, dev=249, test=250
```

## V3 推荐运行顺序

先跑 canonical seed 42：

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\scripts\run_multi_seed_nllb.py --seeds 42
```

对 seed 42 做 bootstrap significance：

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\scripts\bootstrap_significance.py --seeds 42
```

再跑其它 seeds：

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\scripts\run_multi_seed_nllb.py --seeds 13 2026
```

对全部 seeds 汇总 bootstrap 和稳定性：

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\scripts\bootstrap_significance.py --seeds 42 13 2026
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\scripts\summarize_seed_stability.py --seeds 42 13 2026
```

## V3 输出

- `runs/nllb_seed_0042/`
- `runs/nllb_seed_0013/`
- `runs/nllb_seed_2026/`
- `analysis/bootstrap_significance/bootstrap_summary.csv`
- `analysis/bootstrap_significance/bootstrap_summary.md`
- `analysis/seed_stability/seed_stability_summary.csv`
- `analysis/seed_stability/seed_stability_summary.md`

## V3 当前结果摘要

多 seed 稳定性结果如下，数值为 `mean ± sample std`：

| model_id | n_seeds | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1_general_translation | 3 | 4.8841 ± 0.0000 | 25.9489 ± 0.0000 | 95.8452 ± 0.0000 | 12.3529 ± 0.0000 | 16.5025 ± 0.0000 |
| B2_domain_adapted | 3 | 17.4398 ± 0.4606 | 44.5986 ± 0.1064 | 68.3239 ± 0.3095 | 31.7647 ± 1.0189 | 44.5813 ± 1.4982 |
| B3_entity_aware | 3 | 18.3225 ± 0.0000 | 43.3200 ± 0.0000 | 75.8168 ± 0.0000 | 56.6176 ± 0.0000 | 71.9212 ± 0.0000 |
| Ours_entity_aware_domain_adapted | 3 | 31.2881 ± 0.0502 | 52.6101 ± 0.1331 | 55.4214 ± 0.3503 | 60.7843 ± 0.3061 | 77.1757 ± 0.6198 |

Bootstrap significance 使用 5000 次 paired bootstrap resampling。三个 seed 中，Ours 相比 B2 和 B3 在 BLEU、chrF、TER、TA、TCR 上均达到 `p < 0.05`。其中 TER 的 delta 定义为 `baseline TER - Ours TER`，因此正值表示 Ours 更好。

seed 42 的核心显著性结果：

| comparison | metric | observed_delta | 95% CI | p_value |
| --- | --- | ---: | --- | ---: |
| Ours vs B2 | BLEU | 14.2256 | [10.9881, 17.6234] | 0.0002 |
| Ours vs B2 | chrF | 8.1354 | [5.8140, 10.4897] | 0.0002 |
| Ours vs B2 | TER | 12.7841 | [9.8412, 15.9309] | 0.0002 |
| Ours vs B2 | TA | 29.8529 | [25.4927, 34.2593] | 0.0002 |
| Ours vs B2 | TCR | 33.7438 | [25.0909, 36.2458] | 0.0002 |
| Ours vs B3 | BLEU | 13.0108 | [10.7599, 15.7096] | 0.0002 |
| Ours vs B3 | chrF | 9.2918 | [7.6105, 11.0481] | 0.0002 |
| Ours vs B3 | TER | 20.6321 | [15.1971, 27.2598] | 0.0002 |
| Ours vs B3 | TA | 3.8235 | [1.6081, 6.2885] | 0.0006 |
| Ours vs B3 | TCR | 4.6798 | [1.2431, 7.2861] | 0.0012 |
