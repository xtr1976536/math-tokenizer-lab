# GRE 合并 benchmark

`benchmark.jsonl` 是外部论文公式与原 1500 条中通过 GRE 认证且不重复的分量的合并版。每条记录保留来源、原式、参数族信息、GRE 序列和转换证据。

当前共 1252 个去重结构，均位于现有 57-ID GRE 词表内。外部论文参数族实例单独标记，不能当作 1252 篇论文或 1252 个独立物理方程。需要用于当前 EqGPT 50-token 设置时，请使用外部 benchmark 的 `benchmark_maxpos50.jsonl`；合并版目前按结构保存，长度筛选字段见每条记录。

原始 1500 条的完整逐条结论在 `data/formulas/pde1500_gre_review/`，新的来源与范围说明在 `reports/gre_benchmark/结果与范围.md`。
# GRE 合并 benchmark

本目录是由外部论文公式和原 1500 条 PDE 中通过 GRE 认证、去重后的分量组成的 GRE 合并资源；它不是本项目新收集外部 benchmark 的原始独立清单，也不是下游训练池。该合并资源与 EqGPT 的 221 条手册数据、独立的新外部 benchmark 是三个不同用途的文件集，不应混称。

本目录 `benchmark.jsonl` 当前包含 1,252 个去重 GRE 结构。新外部 benchmark 的独立过滤清单（2,045 条）位于 [`../gre_benchmark_filtered/`](../gre_benchmark_filtered/)，外部来源采集记录位于 [`../gre_external/`](../gre_external/)。新下游实验从独立外部清单中选入满足共同 80-token 容量的 1,643 条，并与原 EqGPT benchmark 排除八个目标后的 212 条合并，形成 1,855 条训练来源。训练时的确切来源和 32 次配对增强见 [`../expanded_downstream_80_v1/`](../expanded_downstream_80_v1/)。容量筛选不改变任何原始 benchmark 清单。

原 1500 条的完整逐条结论在 `../pde1500_gre_review/`，合并资源的来源与范围说明在 [`../../../reports/gre_benchmark/结果与范围.md`](../../../reports/gre_benchmark/结果与范围.md)。新增外部清单的来源和两轮审计见 [`../../../reports/gre_benchmark/`](../../../reports/gre_benchmark/) 及 `../gre_benchmark_audit_v2/`。新 benchmark 不等于新增测试案例，八个已知案例仍作为独立下游评估集。

每个数据来源需要遵循其原始出版与数据许可；此清单作为研究数据整理物发布，不替代原始来源许可和引用义务。
