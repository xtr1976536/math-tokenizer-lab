# 方案一：三划分上游实验说明

本目录与 `experiments/downstream/` 的原 212 条语料下游配成方案一。方案二是独立的 [8:1:1 上游](../../experiments/upstream_811/README.md) 和 [扩大语料下游](../expanded_downstream_80_v1/README.md)；完整对应见[总索引](../实验方案索引.md)。

主实验比较 Hard AST-BPE、Qwen2.5、Llama 3 和 `o200k_base`。1500 条 PDE 按来源分成 3 份，每份 500 条，每次选其中 2 份作为训练集、1 份作为测试集，共做 3 次。删除与测试集结构重复的训练公式后，三次训练集分别为 999、1000、999 条。另做一组用简单公式训练、复杂公式测试的实验。详细结果和词表在 `experiments/upstream_main/`，汇总见 `main_summary.csv`。

`three_metrics.csv` 和 `usage_summary.csv` 汇总 Hard AST-BPE 在 200–450 ID 的 12 个容量点；`per_token_usage.csv` 记录每个实际输出 token 的训练、测试次数与概率。对应图见 `reports/vocab_usage/`。实验日志位于 `experiments/upstream/`，仅摘录当前方法的运行行，原始日志的其他方法不在本仓库。

三个性能指标：**tokenizer compression** 为基础字节序列长度除以最终 token 序列长度；**operation** 是完整 LaTeX 命令不被内部切开的比例（旧称 Cmd）；**numbers** 是数字串不被内部切开的比例（旧称 Num）。频率分布用于词表诊断，不与三项性能指标混算。历史逐公式 JSON 仍保留 `compression_ratio`、`cmd_pct`、`num_pct` 字段供复核；对外汇总 CSV 已使用新名称。
