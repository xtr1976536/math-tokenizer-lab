# 上游实验说明

主实验只含 Hard AST-BPE、Qwen2.5、Llama 3 和 `o200k_base`。每一种划分的逐公式结果与训练出的 Hard 词表在 `experiments/pde1500_main_final_20261022/`；`main_summary.csv` 是四种划分的汇总。

`three_metrics.csv` 和 `usage_summary.csv` 汇总 Hard AST-BPE 在 200–450 ID 的 12 个容量点；`per_token_usage.csv` 记录每个实际输出 token 的训练、测试次数与概率。对应图见 `reports/pde1500_vocab_200_450_20261022/`。实验日志位于 `experiments/upstream/`，仅摘录当前方法的运行行，原始日志的其他方法不在本仓库。

三个性能指标：**tokenizer compression** 为基础字节序列长度除以最终 token 序列长度；**operation** 是完整 LaTeX 命令不被内部切开的比例（旧称 Cmd）；**numbers** 是数字串不被内部切开的比例（旧称 Num）。频率分布用于词表诊断，不与三项性能指标混算。历史逐公式 JSON 仍保留 `compression_ratio`、`cmd_pct`、`num_pct` 字段供复核；对外汇总 CSV 已使用新名称。
