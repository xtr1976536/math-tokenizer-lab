# 已训练权重直接使用

缩小版八案例实验共保存 36 份 Transformer 权重：4 份共享预训练权重和 32 份搜索后权重。Hard AST-BPE 有 280、315、350 ID 三种大小，EqGPT GRE 为 57 ID。文件和张量哈希已与训练、搜索记录核对。

权重在 [公开的 GitHub Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/eqgpt-fast-20261022-weights-v1)，下载后统一放入 `data/weights/`。文件清单见 [`data/weights/manifest.json`](../../data/weights/manifest.json)。请使用仓库最新代码；Release 标签只用于下载权重。

## 1. 安装环境与权重

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-eqgpt.lock.txt
PYTHONPATH=src .venv/bin/python scripts/eqgpt/使用已训练模型.py --download
```

下载无需 GitHub 登录。脚本将 36 个 Release 资产按清单保存到 `data/weights/`，逐一核对文件 SHA-256；已有且正确的文件会跳过。Llama 3 等上游分词器的依赖见根目录 `requirements.lock.txt`。已有的八案例 MLP 拟合结果位于 `data/mlp_cache/<case>/`，随 Git 仓库提供，无须从 Release 下载，也无须再次进行 50,000 次拟合。

## 2. 直接生成与评价

```bash
# 加载八案例搜索后模型，直接生成候选公式
PYTHONPATH=src .venv/bin/python scripts/eqgpt/使用已训练模型.py \
  --case Wave_equation --arm ast_bpe_350 --stage searched --action generate --attempts 10

# 载入已拟合 MLP 和原最小二乘 reward，评价一个 LaTeX 候选
PYTHONPATH=src .venv/bin/python scripts/eqgpt/使用已训练模型.py \
  --case Wave_equation --arm ast_bpe_350 --stage searched --action score --formula 'u_t+u_x=0'

# 从共享预训练权重开始，对现有案例数据继续搜索结构（会有计算开销）
PYTHONPATH=src .venv/bin/python scripts/eqgpt/使用已训练模型.py \
  --case Wave_equation --arm ast_bpe_350 --stage pretrained --action search --attempts 200
```

`--arm eqgpt_reference` 使用缩小版 EqGPT GRE 57-ID 模型。`--device` 可选 `cpu`、`mps`、`cuda`。原始 MLP 缓存在 MPS 上计算；CPU 加载时脚本检查保存的网络权重和坐标哈希，并以数值误差阈值比较预测，而非要求不同设备的浮点输出逐字节一致。

老师研究新案例时，可以先不拟合数据，直接用共享预训练模型生成候选：

```bash
PYTHONPATH=src .venv/bin/python scripts/eqgpt/使用已训练模型.py \
  --case new --arm ast_bpe_350 --stage pretrained --action generate --attempts 20
```

## 3. 新案例的边界

共享预训练模型可用于**生成候选结构**。要对老师自己的新时空数据运行 reward 搜索，仍需先提供该案例的可观测变量、坐标、数据格式，并拟合一次相应的 surrogate；八案例缓存只对应原八个数据集，不能直接当作新案例的观测数据。新案例也必须处于当前 GRE/LaTeX 语法和词表支持范围内。此脚本目前已经在 Wave 案例上实际通过 `generate`、`score` 和小预算 `search` 三种调用；不能将旧案例的成功率外推至新物理系统。
