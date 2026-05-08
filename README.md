# YYQEI 中医术语翻译实验

本仓库当前正式使用的论文实验包是 **V2**。V2 面向中医基础理论相关的中英翻译任务，包含严格去泄漏后的真实数据、实验脚本、模型输出和评价结果。

## 当前版本：V2

- 实验包：[`V2/`](V2/)
- 数据目录：[`V2/YYQEI_20260508_NLLB_ready/`](V2/YYQEI_20260508_NLLB_ready/)
- 严格划分：train 990 / dev 249 / test 250
- 测试集规模：250 条
- 对比模型：`facebook/nllb-200-distilled-600M` 与 `Helsinki-NLP/opus-mt-zh-en`
- 防泄漏策略：模型输入只使用训练集术语；完整术语表仅用于审计和 gold metadata
- 已禁用后处理式 `Terminology: ...` 术语补词
- GitHub 包不包含模型 checkpoint、LoRA 权重和原始 PDF

## 实验结果

以下结果均来自 V2 的同一个 strict/no-leak 测试集。所有设置均满足 `normalization_used=False`。

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

## 快速验证

```powershell
cd V2
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_01_validate.py
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_02_check_env.py
```

运行完整实验：

```powershell
cd V2
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_10_run_nllb.py
& 'D:\Anaconda\Scripts\conda.exe' run -n YYQEI python .\run_11_run_helsinki.py
```

## 版本说明

- [`V2/`](V2/)：当前推荐版本，用于论文实验和后续复现。
- [`V1/`](V1/)：历史版本归档，仅用于追溯早期实验。

后续新实验应以 `V2/` 为准。
