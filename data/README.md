# 数据与已训练模型

语料、八案例原始数据、MLP 缓存和模型权重统一放在这个目录。实验配置、日志和搜索结果保存在根目录的 `experiments/`；统计汇总保存在 `reports/`。

## 目录

| 路径 | 内容 |
|---|---|
| [`corpora/benchmark_pde1500_20261022/`](corpora/benchmark_pde1500_20261022/) | 1500 条 PDE、原有 55,644 条非 PDE 公式、来源与筛查记录。当前上游实验只使用 PDE。 |
| [`corpora/eqgpt/PDE_dataset_0725.xlsx`](corpora/eqgpt/PDE_dataset_0725.xlsx) | EqGPT 原作者的 221 条公式工作簿。 |
| [`corpora/eqgpt/augmented/`](corpora/eqgpt/augmented/) | 八案例下游实验实际使用的增强训练序列及目标排除记录，各案例分别保存。 |
| [`corpora/snapshots/`](corpora/snapshots/) | 两套已完成上游实验读取的 PDE 语料快照。 |
| [`cases/`](cases/) | 八个案例的原始 MAT、CSV、NPY 数据及作者的数据说明。 |
| [`surrogates/`](surrogates/) | 各案例已拟合的 MLP、采样坐标、预测值和拟合质量记录。 |
| [`tokenizers/`](tokenizers/) | Qwen2.5、Llama 3 的分词器文件、来源和许可。 |
| [`weights/manifest.json`](weights/manifest.json) | 36 份 Transformer 权重的下载文件名、保存路径、大小与哈希。 |
| `weights/pretrained/` | 4 份共享预训练权重：Hard 280/315/350 与 EqGPT GRE 57。运行下载脚本后生成。 |
| `weights/searched/<case>/` | 每个案例的 4 份搜索后权重，共 32 份。运行下载脚本后生成。 |

Hard AST-BPE 的冻结词表仍与对应实验一起保存在 `experiments/`，入口见 [根目录 README](../README.md#分词器代码与词表)。

## 八个案例

| 案例 | 原始数据 | 已拟合 MLP |
|---|---|---|
| Wave | [Wave_equation](cases/Wave_equation/) | [缓存](surrogates/Wave_equation/surrogate_cache.pt) |
| Convection-diffusion | [Convection_diffusion_equation](cases/Convection_diffusion_equation/) | [缓存](surrogates/Convection_diffusion_equation/surrogate_cache.pt) |
| Klein-Gordon | [KG_equation](cases/KG_equation/) | [缓存](surrogates/KG_equation/surrogate_cache.pt) |
| Burgers | [Burgers_equation](cases/Burgers_equation/) | [缓存](surrogates/Burgers_equation/surrogate_cache.pt) |
| Eq. 6.2.12 | [Eq_6_2_12](cases/Eq_6_2_12/) | [缓存](surrogates/Eq_6_2_12/surrogate_cache.pt) |
| PDE divide | [PDE_divide](cases/PDE_divide/) | [缓存](surrogates/PDE_divide/surrogate_cache.pt) |
| KdV | [KdV_equation](cases/KdV_equation/) | [缓存](surrogates/KdV_equation/surrogate_cache.pt) |
| Chafee-Infante | [Chaffee_Infante_equation](cases/Chaffee_Infante_equation/) | [缓存](surrogates/Chaffee_Infante_equation/surrogate_cache.pt) |

每个 `surrogates/<case>/` 目录包含：

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

## 来源和旧路径

八案例数据及公式工作簿来自 [EqGPT 原作者仓库](https://github.com/woshixuhao/EqGPT)。固定版本、许可和原始文件哈希见 [`third_party/eqgpt/provenance.json`](../third_party/eqgpt/provenance.json)。原版程序在 [`third_party/eqgpt/code/`](../third_party/eqgpt/code/)，其中包含完整 PINN 流程。

本次只调整存放位置，语料、缓存和权重内容未重算。旧路径对应关系如下：

| 旧位置 | 新位置 |
|---|---|
| `data/benchmark_pde1500_20261022/` | `data/corpora/benchmark_pde1500_20261022/` |
| `third_party/eqgpt/data/` | `data/cases/` |
| `third_party/eqgpt/code/PDE_dataset_0725.xlsx` | `data/corpora/eqgpt/PDE_dataset_0725.xlsx` |
| `experiments/downstream/<case>/source/` | `data/corpora/eqgpt/augmented/<case>/` |
| `experiments/downstream/<case>/surrogate/` 及 `surrogate_quality.json` | `data/surrogates/<case>/` |
| `experiments/<run>/equations.jsonl.gz` | `data/corpora/snapshots/<run>/equations.jsonl.gz` |
| `models/` | `data/tokenizers/` |
| `benchmark/weights_manifest.json` | `data/weights/manifest.json` |
| `release_assets/weights/` | `data/weights/` |

EqGPT 原版代码目录中的数据路径和公式工作簿路径保留相对链接，指向这里的唯一一份文件。原版 Python 源码没有改动。历史日志及配置中记录的运行时路径保留原样，可按上表查找。

八案例用于已知方程结构恢复；未知控制方程的数据尚未纳入。
