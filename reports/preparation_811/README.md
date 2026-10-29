# 方案二的上游报告与准备记录

本目录对应[8:1:1上游](../../experiments/upstream_811/README.md)，与[80-token扩大语料下游](../expanded_downstream_80_v1/README.md)共同组成方案二。方案一的三划分上游和212条下游仍完整保留，见[总索引](../实验方案索引.md)。

[实验结果与限制](实验结果与限制.md)包含基础表、总容量、验证选择和冻结测试结果；[验证汇总](validation_summary.csv)包含132个组合；`checks.json`是当时的独立重算记录。`reproducibility_manifest.json`是准备阶段的代码哈希快照，其中GRE转换器后来经过修正，不能解释为当前代码全部与历史快照相同。当前交付物另由总索引所链接的三轮检查验证。

报告下半部分和日志保留当时1500条公式GRE转换的首轮结果（并非最终新下游语料）。后续组件复核、外部benchmark和实际训练池分别见 `reports/gre_benchmark/`、`data/formulas/gre_benchmark_filtered/`和 `data/formulas/expanded_downstream_80_v1/`。正式下游已完成，旧日志中的“未启动”仅描述准备阶段。
