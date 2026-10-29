# 方案二：80-token模型直接使用

本页仅对应扩大语料方案：1855条源公式、59360个增强样本、25 epoch、3层Transformer、80位置容量。Hard词表来自 `data/formulas/expanded_downstream_80_v1/tokenizers_round_0/`，不能使用方案一下游词表或50位置权重。方案一仍使用[原独立入口](README_权重.md)。

## 安装与下载

先按[环境说明](../../ENVIRONMENT.md)安装依赖，然后从仓库根目录运行：

```bash
PYTHONPATH=.:src python scripts/eqgpt/download_expanded_80_weights.py
PYTHONPATH=.:src python scripts/eqgpt/download_expanded_80_weights.py --verify-only
```

36份权重从[方案二Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/expanded-downstream-80-v1-weights)下载，SHA-256和大小见[清单](../../experiments/expanded_downstream_80_v1/weight_manifest.json)。原八案例MLP由仓库提供，新工具只读加载，禁止自动重新拟合。

## 检查、生成和评分

```bash
# 加载并核验指定权重、张量哈希、80位置与配套词表；只读
PYTHONPATH=.:src python scripts/eqgpt/use_expanded_model.py --action inspect
# 使用Wave搜索后Hard315生成10个候选；不计算reward，不改变任何权重
PYTHONPATH=.:src python scripts/eqgpt/use_expanded_model.py --action generate --attempts 10
# 从同一份共享预训练EqGPT权重生成
PYTHONPATH=.:src python scripts/eqgpt/use_expanded_model.py \
  --arm eqgpt_reference --stage pretrained --action generate --attempts 10
# 用Wave已有MLP与原reward评分LaTeX；是结构评分，不是独立系数拟合
PYTHONPATH=.:src python scripts/eqgpt/use_expanded_model.py \
  --action score --formula 'u_{tt}-u_{xx}=0'
```

`--arm`支持EqGPT和Hard280/315/350；`--case`支持八案例及`new`；`--stage`支持pretrained/searched；设备默认CPU，可选mps/cuda。`--max-tokens`默认80，可在3–80之间选择较小上限，仅影响这次生成/搜索，不修改已完成实验。生成模式不执行搜索时的reward与物理mask更新，所以其输出不应被当作正式搜索轨迹的逐项重放。

新案例只可先生成候选：

```bash
PYTHONPATH=.:src python scripts/eqgpt/use_expanded_model.py \
  --case new --stage pretrained --action generate --attempts 10
```

对新数据评分仍需自己的观测变量、数据处理及一次surrogate拟合。原案例MLP不可用于新数据。

## 在新目录中重新搜索

```bash
PYTHONPATH=.:src python scripts/eqgpt/use_expanded_model.py \
  --case Wave_equation --arm ast_bpe_315 --stage pretrained --action search \
  --attempts 200 --output experiments/new_search/study2_Wave_hard315
```

只允许从pretrained开始，输出目录必须尚不存在，并位于两套已归档实验、原始数据、源码和报告目录之外。仍运行3轮、每轮2epoch continuation；仅新输出目录产生搜索记录，不覆盖已有结果，不保存为另一份正式八案例结论。`--seed`控制新生成/搜索；MLP始终加载原种子525的已拟合缓存。
