# 新外部 PDE Benchmark（独立清单）

本目录保存新收集的外部 PDE benchmark，和 EqGPT 原有的 221 条手册 benchmark 分开维护。GRE 转换并去重后有 2,049 条；再排除 4 条与下游测试目标高度相似的公式后，`benchmark.jsonl` 保留 2,045 条。来源记录见 [`../gre_external/`](../gre_external/)，筛查前后统计和两轮审计见 [`../../../reports/gre_benchmark/`](../../../reports/gre_benchmark/) 及 [`../gre_benchmark_audit_v2/`](../gre_benchmark_audit_v2/)。

该清单中的公式可转换为现有 EqGPT GRE 的 57-ID 结构表示，并按既定规则与原 EqGPT 221 条目标结构排重。它是新的外部训练 benchmark，不是八案例下游评估集。

扩大语料下游实验在共同 80-token 容量下从本清单选入 1,643 条，再与原 benchmark 排除八个目标结构后的 212 条合并为 1,855 条训练来源。确切子集、配对增强文本和两轮训练前检查见 [`../expanded_downstream_80_v1/`](../expanded_downstream_80_v1/)。80 容量过滤只用于该实验，没有删改本目录中的 2,045 条清单。

此清单作为研究数据整理物发布。引用和再利用时仍需遵守每条来源论文、数据库或数据集自身的许可与署名要求。
