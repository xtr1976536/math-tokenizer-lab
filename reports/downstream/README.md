# 下游实验说明

仓库中有两套相互独立的八案例下游实验。旧版在 `experiments/downstream/`，训练语料来自 EqGPT 原有 221 条手册 PDE（逐案例排除目标后 212 条）；扩大语料版在 `experiments/expanded_downstream_80_v1/`，包含 1,643 条新外部 benchmark 公式和相同的 212 条非目标 EqGPT 公式。新版本使用 80 token 容量和重新采样的 32 次等价排列。两版结果不可混合覆盖，且不构成仅改变训练语料大小的单变量对照。新数据组成见 [`data/README.md`](../../data/README.md)，完整逐案例结果和逐项比较见 [`reports/expanded_downstream_80_v1/完整结果.json`](../expanded_downstream_80_v1/完整结果.json) 与 [`comparison_previous_zh.md`](../expanded_downstream_80_v1/comparison_previous_zh.md)。

八案例对照为 Hard AST-BPE 280/315/350 ID 与 EqGPT GRE 57 ID，共 32 个实验 arm。`对比结果.csv` 给出各案例 Top-1、Top-10、候选数量和时间；`完整结果.json` 保留同一批实验的结构化记录；`target_exclusion_audit.json` 核查目标 PDE 结构及仅差系数的同结构公式未进入词表或 Transformer 训练语料。

每个 `experiments/downstream/<case>/` 目录保存三份 Hard 词表、四个模型的训练记录、候选搜索轨迹和日志。目标排除后的训练序列在 `data/formulas/eqgpt/augmented/<case>/`，已拟合 MLP 缓存在 `data/mlp_cache/<case>/`。36 份 Transformer 权重已发布到 [GitHub Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/eqgpt-fast-20261022-weights-v1)，下载后存入 `data/weights/`。路径与使用方法见 [数据说明](../../data/README.md)。

这是比原论文更小的快速实验：3 层、25 epoch、每轮 200 次候选、3 轮、每轮继续训练 2 epoch、种子 525。reward 中的最小二乘仍使用原作者实现；最终 PINN 不运行。因此结果仅支持结构命中结论，不支持系数精度或多种子成功率结论。
