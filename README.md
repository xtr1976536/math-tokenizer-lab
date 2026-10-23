# 数学公式分词与 PDE 方程发现

本仓库只保留四块内容：**上游分词实验、下游方程发现实验、实验 benchmark、参考资料**。仓库目前为私有。[Hard AST-BPE 分词可视化](docs/index.html)可以查看输入公式的合并过程，也可以从 1500 条 PDE 公式中选择案例。

## 1. 上游任务

比较 **Hard AST-BPE、Qwen2.5、Llama 3、`o200k_base`** 四个分词器。最新主实验为 1500 条 PDE 的三折测试和“简单训练→复杂测试”；补充实验扫描 Hard AST-BPE 的 200–450 总词表规模。主要指标只有压缩率、完整 LaTeX 命令比例（Cmd）和数字串完整比例（Num），另保留 token 使用次数及概率。

- [主实验结果](reports/upstream/main_summary.csv)、[词表规模指标](reports/upstream/three_metrics.csv)、[逐 token 使用统计](reports/upstream/per_token_usage.csv)
- [主实验逐公式记录与冻结词表](experiments/pde1500_main_final_20261022/)、[Hard 词表扫描](experiments/pde1500_sweep_200_450_20261022/)
- [主实验日志](experiments/upstream/pde1500_main_final_20261022.log)、[扫描日志](experiments/upstream/pde1500_sweep_200_450_20261022.log)（均从同批原始日志提取本仓库当前方法的行）
- [四分词器加载脚本](scripts/当前四分词器.py)、[通用词表文件](models/)、[Hard 实现](src/math_tokenizer/hard_ast_2026.py)

调用示例：`PYTHONPATH=src python scripts/当前四分词器.py hard_ast_bpe 'u_t+u_x=0'`。其余方法名为 `qwen2.5`、`llama3` 和 `o200k_base`。网页只展示 Hard AST-BPE。

重新评估固定公式集可运行 `PYTHONPATH=src:. python scripts/evaluate_screened_pde.py --output experiments/复核主实验 --sizes 256`。已有实验记录不需为浏览网页重新计算。

## 2. 下游任务

比较 Hard AST-BPE（280、315、350 ID）与 **EqGPT GRE（57 ID）** 接入同一个缩小后的方程发现模型，在八个已知 PDE 案例中搜索正确方程结构。两个模型均从头训练；每个案例共用一个预先拟合并缓存的 MLP surrogate，保留 EqGPT 原最小二乘 reward、候选排序和 Top-10 继续训练。末尾 PINN 系数拟合没有运行，因此这里比较的是**公式结构发现**，不是系数精度。

| 参数 | 本仓库快速实验 | EqGPT 原论文/作者默认设置 |
|---|---:|---:|
| Transformer 层数 | **3** | 6 |
| 模型宽度、注意力头数 | 768、8 | 768、8 |
| 初始训练 | **25 epoch** | 100 epoch |
| 候选搜索 | **3 轮 × 每轮 200 次** | 5 轮 × 每轮 400 次 |
| 每轮继续训练 | **2 epoch** | 5 epoch |
| 随机种子 | 525（1 个） | 依论文具体实验而定 |
| surrogate | 5 层、宽 50 的 MLP，拟合 50,000 次 | 原作者数值路径 |
| 最终 PINN | 不运行 | 原论文使用 |

实际参数以[保存的运行协议](experiments/downstream/Wave_equation/protocol.json)为准；这是**缩小后的对照实验**，不能等同于原论文的完整预算。[Hard 与 GRE 对比结果](reports/downstream/对比结果.csv)、[八案例完整搜索记录](experiments/downstream/)和[目标公式排除审计](reports/downstream/target_exclusion_audit.json)均已保存。每个案例的运行日志从同批原始日志中提取了共享 surrogate、Hard 和 GRE 的行；结果文件只收录 Hard 和 GRE。

## 3. Benchmark

- [上游固定公式集](data/benchmark_pde1500_20261022/)：1500 条 PDE 来自 490 篇论文，简单/中等/复杂按语法结构为 450/600/450；另有 55,644 条原有非 PDE 公式记录。质量检查包括来源回放、解析检查和语义抽查，**不是**专家逐条认证。
- [下游八案例数据](third_party/eqgpt/data/)与[EqGPT 原作者 PDE 结构工作簿](third_party/eqgpt/code/PDE_dataset_0725.xlsx)：八案例是**已知方程回收测试**，不是从未知真实过程发现新 PDE 的 benchmark。

此前讨论的真实数据新方程 benchmark 尚未定稿，**不在本仓库**。

## 4. 参考资料

[参考论文 PDF、原仓库地址与许可说明](参考资料/README.md)。第三方原始源码保持原貌；本仓库说明文字先用中文，待内容确认后再制作英文版本。

网页可以直接打开 [docs/index.html](docs/index.html)，无需启动服务。GitHub API 返回“当前套餐不支持为这个私有仓库启用 Pages”，所以网页目前已入仓库、可本地预览，但未形成 Pages 地址。
