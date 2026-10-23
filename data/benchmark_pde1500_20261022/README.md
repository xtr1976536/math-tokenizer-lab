# 固定公式 benchmark

`pde_equations.jsonl.gz`：1500 条 PDE 公式，来自 490 个来源组，按语法结构复杂度为简单 450、中等 600、复杂 450。`formulas.jsonl.gz` 同时保留原有 55,644 条非 PDE 公式记录；当前三折与“简单训练→复杂测试”的分词评估仅使用 PDE 子集。

`manifest.json` 固定数量、来源和哈希；`quality_audit.json`、`semantic_spotcheck.jsonl` 与 `rejections.jsonl` 记录筛查过程。质量检查属于自动验证和语义抽查，不意味着每条公式经 PDE 专家逐一确认。

下游八案例原始数据、来源描述和 EqGPT 原作者的 PDE 结构工作簿在 `third_party/eqgpt/`。这些是已知方程回收实验；未知真实控制方程的数据 benchmark 不在本次仓库中。
