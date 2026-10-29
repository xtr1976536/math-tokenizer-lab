# 数据与已训练模型

语料、八案例原始数据、MLP 缓存和模型权重统一放在这个目录。实验配置、日志和搜索结果保存在根目录的 `experiments/`；统计汇总保存在 `reports/`。

两套完整方案见[实验总索引](../reports/实验方案索引.md)。上游均使用完整 1500 条 PDE；公式 benchmark 的目标排除只用于下游，不用于删减上游 8:1:1 语料。

## 目录

| 路径 | 内容 |
|---|---|
| [`formulas/pde/`](formulas/pde/) | 1500 条 PDE、原有 55,644 条非 PDE 公式、来源与筛查记录。当前上游实验只使用 PDE。 |
| [`formulas/eqgpt/PDE_dataset_0725.xlsx`](formulas/eqgpt/PDE_dataset_0725.xlsx) | EqGPT 原作者的 221 条公式。 |
| [`formulas/eqgpt/augmented/`](formulas/eqgpt/augmented/) | 八案例下游实验实际使用的增强训练序列及目标排除记录，各案例分别保存。 |
| [`formulas/gre_benchmark_filtered/`](formulas/gre_benchmark_filtered/) | 新的独立外部 PDE benchmark（2,045 条）；来源采集记录位于 `gre_external/`。 |
| [`formulas/expanded_downstream_80_v1/`](formulas/expanded_downstream_80_v1/) | 新实验实际使用的 1,855 条共同训练来源及 59,360 条配对增强样本；含原 212 条非目标 EqGPT 公式和 1,643 条新 benchmark 公式。 |
| [`formulas/experiment_inputs/`](formulas/experiment_inputs/) | 各次上游实验使用的公式。 |
| [`cases/`](cases/) | 八个案例的原始 MAT、CSV、NPY 数据及作者的数据说明。 |
| [`mlp_cache/`](mlp_cache/) | 各案例已拟合的 MLP、采样坐标、预测值和拟合质量记录。 |
| [`tokenizers/`](tokenizers/) | Qwen2.5、Llama 3 的分词器文件、来源和许可。 |
| [`weights/manifest.json`](weights/manifest.json) | 36 份 Transformer 权重的下载文件名、保存路径、大小与哈希。 |
| `weights/pretrained/` | 4 份共享预训练权重：Hard 280/315/350 与 EqGPT GRE 57。运行下载脚本后生成。 |
| `weights/searched/<case>/` | 每个案例的 4 份搜索后权重，共 32 份。运行下载脚本后生成。 |

训练好的 Hard AST-BPE 词表仍与对应实验一起保存在 `experiments/`，入口见 [根目录 README](../README.md#分词器代码与词表)。

## 八个案例

| 案例 | 原始数据 | 已拟合 MLP |
|---|---|---|
| Wave | [Wave_equation](cases/Wave_equation/) | [缓存](mlp_cache/Wave_equation/surrogate_cache.pt) |
| Convection-diffusion | [Convection_diffusion_equation](cases/Convection_diffusion_equation/) | [缓存](mlp_cache/Convection_diffusion_equation/surrogate_cache.pt) |
| Klein-Gordon | [KG_equation](cases/KG_equation/) | [缓存](mlp_cache/KG_equation/surrogate_cache.pt) |
| Burgers | [Burgers_equation](cases/Burgers_equation/) | [缓存](mlp_cache/Burgers_equation/surrogate_cache.pt) |
| Eq. 6.2.12 | [Eq_6_2_12](cases/Eq_6_2_12/) | [缓存](mlp_cache/Eq_6_2_12/surrogate_cache.pt) |
| PDE divide | [PDE_divide](cases/PDE_divide/) | [缓存](mlp_cache/PDE_divide/surrogate_cache.pt) |
| KdV | [KdV_equation](cases/KdV_equation/) | [缓存](mlp_cache/KdV_equation/surrogate_cache.pt) |
| Chafee-Infante | [Chaffee_Infante_equation](cases/Chaffee_Infante_equation/) | [缓存](mlp_cache/Chaffee_Infante_equation/surrogate_cache.pt) |

每个 `mlp_cache/<case>/` 目录包含：

- `surrogate_cache.pt`：MLP 参数、供 reward 求导的 10,000 个采样坐标及预测值。
- `shared_manifest.json`：网络配置、数据哈希与预处理记录。
- `surrogate_quality.json`：原拟合质量记录和缓存文件哈希。

这些文件对应已完成的拟合，可以直接复用。新案例仍需用自己的观测数据拟合 surrogate。

## 下载权重

在已安装下游依赖的 Python 环境中，从仓库根目录运行：

```bash
PYTHONPATH=src python scripts/eqgpt/使用已训练模型.py --download
```

脚本从 [公开的 GitHub Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/eqgpt-fast-20261022-weights-v1) 下载权重，无需 GitHub 登录。已有且哈希正确的文件会跳过。文件下载后按清单放入 `data/weights/pretrained/` 和 `data/weights/searched/`；这两个目录由 Git 忽略，避免将约 2.6 GB 的权重重复提交进源码仓库。

生成候选、计算 reward 和继续搜索的命令见 [使用已训练模型](../scripts/eqgpt/README_权重.md)。

## 来源

八案例数据及公式表来自 [EqGPT 原作者仓库](https://github.com/woshixuhao/EqGPT)。版本、许可和原始文件校验值见 [`eqgpt_original/provenance.json`](../eqgpt_original/provenance.json)。原版程序在 [`eqgpt_original/code/`](../eqgpt_original/code/)，包含完整 PINN 流程。

### 两套 PDE benchmark 的区别

仓库保留两个用途相关、来源不同的公式 benchmark，二者均可转换为 GRE 结构，但不能视为同一数据集的两个名字：

| 名称 | 来源与规模 | 在下游实验中的用途 |
|---|---|---|
| 原 EqGPT benchmark | 数学手册《Nonlinear Partial Differential Equations for Scientists and Engineers》中的 221 条 PDE；每个测试案例排除目标结构及同结构变体后，旧实验使用 212 条。 | `experiments/downstream/` 的旧版训练语料。 |
| 新外部 benchmark | 从论文、公开数据和 PDE 发现 benchmark 收集、转换并审查的公式；筛查后 2,045 条，80 容量共同筛选后纳入新训练池 1,643 条。 | 与原 EqGPT 排除目标后的 212 条合并，构成新实验 1,855 条来源。 |

新外部 benchmark 的独立清单和结构检查见 [`gre_benchmark_filtered/`](formulas/gre_benchmark_filtered/) 及 [`gre_benchmark_audit_v2/`](formulas/gre_benchmark_audit_v2/)。新下游实际训练子集、来源 ID 和增强配对见 [`expanded_downstream_80_v1/`](formulas/expanded_downstream_80_v1/)。80 容量筛选只决定本实验的训练子集；它没有删除或改写新 benchmark 原始清单，也没有修改旧 221 条 benchmark。新训练子集完整排除八个测试目标及审查出的同结构目标，审计细节见其 `audit_pass1.json`、`audit_pass2.json`。

原版代码需要的数据路径和公式表路径保留了相对链接，实际文件都在 `data/`。历史日志中的旧目录名保留原样。

八案例用于已知方程结构恢复，未知控制方程的数据尚未纳入。
