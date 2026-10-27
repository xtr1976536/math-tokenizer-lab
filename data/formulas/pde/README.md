# 上游公式

`pde_equations.jsonl.gz` 保存 1500 条 PDE，来自 490 个来源组。按公式结构分为简单、中等、复杂，分别有 450、600、450 条。

`formulas.jsonl.gz` 还保留原有的 55,644 条非 PDE 公式。当前上游实验只使用 PDE。

实验将 PDE 按来源分成 3 份，每次选其中 2 份作为训练集、1 份作为测试集，共做 3 次。另外做一组用简单公式训练、复杂公式测试的实验。

`manifest.json` 记录公式数量、来源和校验值。`quality_audit.json`、`semantic_spotcheck.jsonl` 和 `rejections.jsonl` 保存自动检查、人工抽查和剔除记录，尚未逐条经过专家审核。

下游八案例的数据在 `data/cases/`，EqGPT 原作者的公式表在 `data/formulas/eqgpt/`。这些案例用于恢复已知方程。
