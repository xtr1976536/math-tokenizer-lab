# 数学公式分词与 PDE 方程发现

我们研究数学公式分词器能否改善 PDE 方程发现的效果。仓库包含 Hard AST-BPE 分词器、上游分词实验，以及接入 EqGPT 的下游实验。下文的 equation-BPE 指 Hard AST-BPE。

## 目录用途

| 文件夹 | 用途 | 主要看什么 |
|---|---|---|
| [`src/math_tokenizer/`](src/math_tokenizer/) | 项目的主要 Python 代码，包括公式解析、Hard AST-BPE、评估，以及下游模型接入。 | 分词算法从 `hard_ast_2026.py` 开始；基础字符表压缩和词表加载在 `eqgpt/compact_alphabet.py`。 |
| [`native/`](native/) | Python 调用的 C 代码，用来加速公式解析、分词和解码。 | `hard_ranked_2026.c` 执行 Hard AST-BPE 的合并；`frontend_v6.c` 处理公式结构。 |
| [`scripts/`](scripts/) | 运行实验、统计结果和使用已训练模型的入口。 | 上游实验看 `evaluate_screened_pde.py`；分词示例看 `当前四分词器.py`；下游实验和权重加载看 `eqgpt/`。 |
| [`data/`](data/README.md) | 集中保存语料、八案例数据、MLP 缓存、通用分词器文件和模型权重。 | 各类数据的路径、来源和下载方法见下表及数据说明。 |
| [`experiments/`](experiments/) | 保存实验运行的详细记录。 | 上游有逐公式结果、训练好的 Hard 词表和日志；下游有各案例的配置、训练记录、候选搜索结果和词表。 |
| [`reports/`](reports/) | 保存整理后的实验结果，方便查看和比较。 | 上游指标汇总、下游对比表、可读词表，以及 token 使用频率和长度统计图。 |
| [`third_party/eqgpt/`](third_party/eqgpt/) | 保存 EqGPT 原版代码、许可和来源说明。 | `code/` 包含原版 PINN 流程；本项目修改后的实现放在 `src/` 和 `scripts/`。 |
| [`参考资料/`](参考资料/) | 保存参考论文和参考仓库链接。 | `PDF/` 是论文文件，`README.md` 说明来源和链接。 |

数据统一放在 `data/` 下：

```text
data/
├── corpora/
│   ├── benchmark_pde1500_20261022/ # 上游公式、来源和筛查记录
│   ├── eqgpt/                     # EqGPT 公式工作簿及八案例的增强训练语料
│   └── snapshots/                 # 已完成上游实验使用的 PDE 语料快照
├── cases/                         # 八个测试案例的原始 MAT、CSV、NPY 数据
├── surrogates/                    # 八案例已拟合 MLP、采样坐标和质量记录
├── tokenizers/                    # Qwen2.5、Llama 3 的分词器文件及来源
└── weights/
    ├── manifest.json              # 36 份权重的文件名、大小和哈希
    ├── pretrained/                # 4 份共享预训练权重，下载后生成
    └── searched/                  # 8 案例 × 4 模型的搜索后权重，下载后生成
```

语料、原始数据和 MLP 缓存随仓库提供。约 2.6 GB 的 Transformer 权重保存在 [GitHub Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/eqgpt-fast-20261022-weights-v1)，运行下载脚本后放入 `data/weights/`。见 [数据与权重说明](data/README.md)。

`experiments/` 与 `reports/` 的主要子目录如下：

| 子目录 | 内容 |
|---|---|
| `experiments/pde1500_main_final_20261022/` | 上游主实验，按三折和“简单训练→复杂测试”保存指标、逐公式结果和 256-ID 词表。 |
| `experiments/pde1500_sweep_200_450_20261022/` | 上游词表大小实验，保存各容量下的结果、词表及 token 使用统计。 |
| `experiments/upstream/` | 上游实验日志。 |
| `experiments/downstream/` | 八个下游测试案例各有一个目录，保存运行配置、Hard 与 GRE 结果、日志和词表。 |
| `reports/upstream/` | 四个分词器的结果，以及 Hard AST-BPE 的词表规模和使用频率汇总。 |
| `reports/downstream/` | Hard 与 GRE 的八案例对比、训练目标排除审计，以及 280、315、350-ID 的可读词表。 |
| `reports/pde1500_vocab_200_450_20261022/` | 词表大小与 token 使用情况的分析表、统计图。 |

## 分词器代码与词表

Hard AST-BPE 的训练、编码和解码在 [`hard_ast_2026.py`](src/math_tokenizer/hard_ast_2026.py) 的 `HardASTBPE` 类中。训练调用 [`atoms_v4.py`](src/math_tokenizer/atoms_v4.py) 中的 `GrammarBPE`；公式结构由 [`frontend_v6.py`](src/math_tokenizer/frontend_v6.py) 解析。编码时使用 [`hard_ranked_2026.c`](native/hard_ranked_2026.c) 执行合并。

[`compact_alphabet.py`](src/math_tokenizer/eqgpt/compact_alphabet.py) 负责压缩基础字符表和映射 token ID。下游通过 [`latex_tokenizers.py`](src/math_tokenizer/eqgpt/latex_tokenizers.py) 中的 `LatexAdapter` 接入 Transformer。

训练好的词表保存在实验目录：

- **上游主实验，256 ID**：[`experiments/pde1500_main_final_20261022/`](experiments/pde1500_main_final_20261022/)，按三折和“简单训练→复杂测试”分别保存。
- **上游词表大小实验，200–450 ID**：[`experiments/pde1500_sweep_200_450_20261022/`](experiments/pde1500_sweep_200_450_20261022/)。
- **下游，280、315、350 ID**：各案例的 `tokenizers/` 目录，例如 [`Wave_equation/tokenizers/`](experiments/downstream/Wave_equation/tokenizers/)。
- **逐项查看词表**：[280 ID](reports/downstream/vocabulary_280.md)、[315 ID](reports/downstream/vocabulary_315.md)、[350 ID](reports/downstream/vocabulary_350.md)。上游各实验目录中的 `*.tokens.csv` 还记录 token 的使用次数和频率。

每份词表需要两个文件：`ast_bpe_<size>.model.json` 保存基础字符表和 ID 数量，`ast_bpe_<size>.model.backend.json` 保存 token 字节和合并规则。加载时两个文件须放在同一目录。

现有实验使用 98 个基础字节字符。总词表大小包含三个控制 ID：`PAD=0`、`EOS=1`、`BOS=2`，因此 256-ID 词表有 253 个内容 token。底层 backend 还保留部分内部字节条目，其条目数不能直接作为最终词表大小。

在已安装依赖的环境中，从仓库根目录运行 Python，并设置 `PYTHONPATH=src`：

```python
from pathlib import Path
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet

path = Path("experiments/pde1500_main_final_20261022/pde_fold_1/ast_bpe_256.model.json")
model = CompactAlphabet.load(path)

# 按下游模型的 ID 顺序打印词表。
for token_id, token in enumerate(("<PAD>", "<EOS>", "<BOS>")):
    print(token_id, token)
for content_id, token_bytes in enumerate(model.vocab):
    print(content_id + 3, repr(token_bytes.decode("utf-8", errors="backslashreplace")))

# encode 返回内容 ID；接入下游模型时，LatexAdapter 会将它们加 3。
formula = "u_t+u_x=0"
ids = model.encode(formula)
print(ids)
print(model.decode(ids))
assert model.decode(ids) == formula
```

首次加载需要 C 编译器。输入超出基础字符表时会报错。不同实验的模型权重应与各自的词表配套使用。

## 上游实验

比较 Hard AST-BPE、Qwen2.5、Llama 3 和 `o200k_base` 四个分词器。实验只使用 1500 条 PDE，采用三折划分和“简单训练→复杂测试”，另测试 Hard AST-BPE 在 200–450 ID 下的表现。

三个指标为 **tokenizer compression、operation、numbers**，分别衡量序列压缩、LaTeX 命令完整性和数字串完整性。定义见 [上游实验说明](reports/upstream/README.md)。我们也统计 token 的使用频率和长度。

[主实验结果](reports/upstream/main_summary.csv) · [词表大小实验](reports/upstream/three_metrics.csv) · [token 使用统计](reports/upstream/per_token_usage.csv) · [日志](experiments/upstream/)

查看分词结果：

```bash
PYTHONPATH=src python scripts/当前四分词器.py hard_ast_bpe 'u_t+u_x=0'
```

其余方法名为 `qwen2.5`、`llama3` 和 `o200k_base`。实验入口为 [`evaluate_screened_pde.py`](scripts/evaluate_screened_pde.py)，词表统计入口为 [`analyze_pde_vocab_usage.py`](scripts/analyze_pde_vocab_usage.py)。

## 下游实验

将 Hard AST-BPE（280、315、350 ID）和 EqGPT GRE（57 ID）分别接入从头训练的 GPT-2 架构模型，在八个已知 PDE 案例上比较方程结构发现结果。

这组实验缩小了原 EqGPT 的训练与搜索规模：Transformer 从 6 层改为 3 层，初始训练从 100 epoch 改为 25 epoch；搜索从 5 轮、每轮 400 次改为 3 轮、每轮 200 次；每轮继续训练从 5 epoch 改为 2 epoch。模型宽度为 768，注意力头数为 8，随机种子为 525。运行配置见各案例的 `protocol.json`，例如 [Wave 配置](experiments/downstream/Wave_equation/protocol.json)。

同一案例的各模型共用已拟合的 MLP surrogate。搜索保留原最小二乘 reward、候选排序和 Top-10 继续训练，不运行末尾的 PINN 系数拟合；本轮评价方程结构的恢复情况。训练语料的目标公式排除记录见 [审计文件](reports/downstream/target_exclusion_audit.json)。

[结果汇总](reports/downstream/对比结果.csv) · [逐案例结果和日志](experiments/downstream/) · [运行脚本](scripts/eqgpt/fast_reward_case.py)

已训练的 36 份 Transformer 权重在 [GitHub Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/eqgpt-fast-20261022-weights-v1)。使用方法见 [模型加载说明](scripts/eqgpt/README_权重.md)，无需重新训练已有案例的 MLP。

## 数据与参考资料

[上游公式集](data/corpora/benchmark_pde1500_20261022/) 包含 1500 条 PDE，来自 490 个来源组，简单、中等、复杂各有 450、600、450 条。目录还保留 55,644 条非 PDE 数学公式，当前上游实验未使用这些公式。筛查记录包括来源、自动检查和语义抽查，尚未逐条经过专家审核。

下游八个测试案例的原始数据在 [`data/cases/`](data/cases/)，已拟合 MLP 在 `data/surrogates/<case>/surrogate_cache.pt`。具体路径见 [数据索引](data/README.md)。这些案例用于已知方程的结构恢复，仓库尚未包含未知控制方程的发现实验。

[参考资料](参考资料/README.md) 收录论文 PDF 和参考仓库链接。[EqGPT 论文原版说明](third_party/eqgpt/论文原版说明.md) 对应作者的完整代码，与本项目的缩小版实验分开保存。

依赖见 [`requirements.lock.txt`](requirements.lock.txt) 和 [`requirements-eqgpt.lock.txt`](requirements-eqgpt.lock.txt)，项目配置见 [`pyproject.toml`](pyproject.toml)。许可与第三方材料说明见 [`LICENSE`](LICENSE) 和 [`NOTICE.md`](NOTICE.md)。
