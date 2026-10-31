# Study 3：语料规模效应

这是第三套正式下游实验，与 `experiments/downstream/`（实验一）和 `experiments/expanded_downstream_80_v1/`（实验二）并列保存。它比较 212 条小语料与 1,100 条大语料在固定八案例 PDE 结构发现任务上的表现，并记录正确结构是在生成、评分还是 Top-10 选择阶段丢失。

## 已完成运行

| 组别 | 未增强源公式 | 配对增强样本 | Hard 词表 | Transformer/搜索容量 | 结果 |
|---|---:|---:|---|---:|---|
| small | 212 | 6,784 | 本组独立训练，280/315/350 ID | 60 | 4 个 arm、32 个搜索组完成 |
| large | 1,100 | 35,200 | 本组独立训练，280/315/350 ID | 60 | 4 个 arm、32 个搜索组完成 |

两组都使用 GRE 统一渲染的未增强文本训练 Hard 词表；Transformer 使用本组 32 条增强序列，EOS 固定在句尾。训练为 25 epoch，搜索为 3 轮×200 候选，每轮 continuation 2 epoch，seed 525。两组共 64 个搜索组、38,400 个候选。八份既有 MLP surrogate 只读复用。

完整结果见 [`reports/study3_corpus_effect_v1/study3_final_report_zh.md`](../../reports/study3_corpus_effect_v1/study3_final_report_zh.md)，逐案例机制记录在 [`reports/study3_corpus_effect_v1/case_mechanism.csv`](../../reports/study3_corpus_effect_v1/case_mechanism.csv)。每个搜索组的 `search.json`、三轮 trace、`mechanism.json`、`diagnostics.json` 和 `complete.json` 均保留在本目录。

## 权重

`.pt` 文件不进入 Git 历史。当前目录中的 328 个本地 checkpoint 路径包含 200 个去重文件和硬链接别名；去重文件发布在 [Study 3 weights Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/study3-corpus-effect-v1-weights)。`weight_manifest.json` 记录 Release asset、目标路径、字节数和 SHA-256，下载与校验：

```bash
PYTHONPATH=src python scripts/eqgpt/download_study3_weights.py
PYTHONPATH=src python scripts/eqgpt/download_study3_weights.py --verify-only
```

manifest 只发布共享预训练权重与每个搜索组的 `cycle_1.pt`、`cycle_2.pt`、`cycle_3.pt`。`cycle_0.pt` 和 `searched.pt` 是本地硬链接别名，避免重复上传；逐组的 `SEARCH_INPUT.json`、`search.json` 和 `complete.json` 记录了它们的哈希关系。

## 复现与审计

```bash
PYTHONPATH=src python scripts/eqgpt/run_study3.py --phase plan
PYTHONPATH=src python scripts/audit_study3.py --mode data --output /tmp/study3_data_check.json
# 运行前集成检查的既有结果：reports/study3_corpus_effect_v1/integration.json
```

正式训练/搜索需要显式 `--execute-reviewed-protocol`，并由当前代码和语料哈希门控。四轮检查、第五轮补充复核、准备和运行日志集中在 `reports/study3_corpus_effect_v1/`。`superseded_50_shared_vocab/` 是早期 50 容量共享词表草稿，明确不属于本次正式结果。

历史准备稿、中断现场与启动前检查的适用范围见[归档边界说明](../../reports/study3_corpus_effect_v1/README_归档边界.md)。
