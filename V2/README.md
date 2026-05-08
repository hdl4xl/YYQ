# YYQEI 2026-05-08 干净实验包

本目录是用于 GitHub 开源和实验复现的 V2 版本包，面向中医基础理论相关的中英翻译实验。

V2 基于 2026-05-08 核验后的真实数据构建，保留实验输入、脚本、评价指标、论文表格和模型翻译输出。该目录不包含训练得到的 LoRA adapter、checkpoint、optimizer 状态、tokenizer 缓存或原始 PDF。

## 包含内容

- `YYQEI_20260508_NLLB_ready/`：strict/no-leak 的 NLLB-ready 真实数据。
- `scripts/`：数据验证和实验脚本。
- `runs/nllb_20260508_strict/`：NLLB 的 prepared 文件、翻译结果和指标。
- `runs/helsinki_20260508_strict/`：Helsinki 的 prepared 文件、翻译结果和指标。
- `docs/`：实验说明和 GitHub clean package 说明。
- `requirements.txt` 与 `environment.yml`：复现实验环境说明。
- `LICENSE` 与 `DATA_LICENSE.md`：代码许可证和数据使用说明。

## 数据划分

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

## 运行完整实验

NLLB-200 distilled 600M：

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_10_run_nllb.py
```

Helsinki OPUS-MT：

```powershell
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_11_run_helsinki.py
```

默认运行环境：

```text
D:\Anaconda\Scripts\conda.exe
conda env: YYQEI
```

## 实验结果

所有结果均来自同一个 V2 strict/no-leak 测试集，测试集为 250 条。所有输出均满足 `normalization_used=False`。

### NLLB-200-distilled-600M

| 实验设置 | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1_general_translation | 4.8841 | 25.9489 | 95.8452 | 12.3529 | 16.5025 |
| B2_domain_adapted | 16.9691 | 43.8895 | 68.8210 | 31.6176 | 44.0887 |
| B3_entity_aware | 18.3225 | 43.3200 | 75.8168 | 56.6176 | 71.9212 |
| Ours_entity_aware_domain_adapted | 31.4978 | 52.5205 | 55.6463 | 60.7353 | 77.0936 |

### Helsinki-NLP/opus-mt-zh-en

| 实验设置 | BLEU | chrF | TER | TA | TCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1_general_translation | 6.7578 | 27.2359 | 83.5227 | 15.8824 | 21.6749 |
| B2_domain_adapted | 13.0818 | 39.5669 | 76.1009 | 30.0000 | 43.3498 |
| B3_entity_aware | 10.2387 | 36.9979 | 73.4730 | 44.2647 | 59.6059 |
| Ours_entity_aware_domain_adapted | 20.1525 | 44.6517 | 64.4886 | 48.2353 | 64.7783 |

在该数据集上，NLLB 的 full method 明显强于 Helsinki；同时两个 backbone 都显示 domain adaptation 与 entity-aware input 对中医术语翻译有明显帮助。

## GitHub 开源边界

本包刻意排除了模型权重和训练状态文件。用户应通过 Hugging Face 自行下载 `facebook/nllb-200-distilled-600M` 和 `Helsinki-NLP/opus-mt-zh-en`，并遵守对应 model card 和许可证。

具体排除策略见 `docs/GITHUB_CLEAN_PACKAGE.md`。
