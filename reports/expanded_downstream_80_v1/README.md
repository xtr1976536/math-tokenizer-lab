# 方案二：扩大语料、80-token下游

本目录与[8:1:1上游](../../experiments/upstream_811/README.md)配成方案二；方案一为三划分上游和212条下游。两套实验文件关系见[总索引](../实验方案索引.md)。

- [实验结果](report_zh.md)：八案例Top1/Top10、冗余、可评估率和记录耗时。
- [逐案例CSV](case_results.csv)与[完整JSON](完整结果.json)：32组结构化汇总，详细候选见[实验目录](../../experiments/expanded_downstream_80_v1/README.md)。
- [与方案一下游的比较](comparison_previous_zh.md)和[逐项CSV](comparison_previous.csv)：明确容量、输入文本、增强和训练量均有变化，非单变量语料量消融。
- [完成核验](verification.json)与[下游仓库三轮检查](repository_checks_3rounds.json)：四份25epoch预训练、32份600候选及权重与MLP缓存核验。
- `logs/pass1.log`、`logs/pass2.log`、`实验设置与检查.md`：正式训练前检查记录；`启动记录.json`为完成状态摘要。

共同训练源1855条（新外部1643＋原EqGPT非目标212），每源32次配对等价排列，共59360样本。Hard使用原LaTeX；GRE使用同源结构。八案例共用已有MLP缓存，未重新拟合，未运行PINN或独立系数拟合。36份权重与下载方法见[实验README](../../experiments/expanded_downstream_80_v1/README.md)。
