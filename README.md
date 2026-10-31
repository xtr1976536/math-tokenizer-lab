# 数学公式分词与 PDE 方程发现

我们研究数学公式分词器能否改善 PDE 方程发现的效果。仓库包含 Hard AST-BPE 分词器、上游分词实验，以及接入 EqGPT 的下游实验。下文的 equation-BPE 指 Hard AST-BPE。

## 三套完整下游实验

仓库保留两套完整“上游＋下游”研究方案，以及第三套独立的下游语料效应实验。三套共用基础代码、八案例数据和既有 MLP；语料、词表、容量、增强协议、权重和结果按方案分开保存。

| 项目 | 实验一：三划分上游＋原 benchmark | 实验二：8:1:1 上游＋扩大语料 | 实验三：语料规模效应 |
|---|---|---|---|
| 上游 | 1500 条 PDE；三组轮换训练/测试及简单到复杂对照 | 完整 1500 条；1200/150/150 分组划分 | 无新增上游实验 |
| 下游训练源公式 | 原 221 条手册，排除目标后 212 条 | 1643 条外部＋原 212 条＝1855 条 | 小组212条；大组1100条（原212＋外部888） |
| Transformer 增强样本 | 6784 条 | 59360 条 | 小组6784；大组35200 |
| Hard 词表训练文本 | 32倍增强后的 GRE 渲染 LaTeX | 未增强原始 LaTeX | 各组分别用本组未增强 GRE 渲染 LaTeX 学习 |
| Hard 词表关系 | 每案例词表，280/315/350 ID | 新语料独立词表，280/315/350 ID | 小、大语料分别重学，容量相同但词表内容/ID映射不同 |
| 增强 | 作者排列增强；原流程 EOS 行为 | 固定种子采样32个等价排列，EOS句尾，保留原始LaTeX | 仅交换加法项与乘法因子；32条配对样本，EOS句尾，统一GRE渲染LaTeX |
| 训练/搜索容量 | 50 token | 80 token | 60 token |
| 下游文件 | [八案例与汇总](reports/downstream/README.md) | [实验说明](experiments/expanded_downstream_80_v1/README.md) | [代码、设置、完整结果](reports/study3_corpus_effect_v1/README_实验协议与检查入口.md) |

三套下游均使用 EqGPT GRE、Hard AST-BPE 280/315/350、3层 Transformer、25 epoch、seed 525、八案例、每案例每arm 3×200 候选、每轮2 epoch continuation，并只读复用同一批八案例 MLP 缓存；均未做最终 PINN 系数拟合。实验三在本次单种子八案例评估中，小语料的 EqGPT/Hard280/315/350 Top1 为1/8、3/8、3/8、3/8；大语料为0/8。实验三的大语料同时改变训练语料数量/组成和 Hard 词表所学内容；25 epoch相同但更新步数不同，因此结果不是固定算力、纯数量的因果效应。

三套实验的详细对照、文件对应、权重和核验记录见[实验总索引](reports/实验方案索引.md)。新外部 benchmark 多数记录是来源明确的参数/函数族实例，不应把来源记录数当成独立论文或独立 PDE 家族数。

## 快速查看与使用

只查看结论时，打开[三套实验索引](reports/实验方案索引.md)、[实验一结果](reports/downstream/README.md)、[实验二结果](reports/expanded_downstream_80_v1/report_zh.md)和[实验三配对报告](reports/study3_corpus_effect_v1/study3_final_report_zh.md)，无需下载权重或重新训练。

安装环境见[环境说明](ENVIRONMENT.md)。方案一模型使用见[50-token模型入口](scripts/eqgpt/README_权重.md)，方案二见[80-token模型入口](scripts/eqgpt/README_方案二模型使用.md)，方案三见[Study 3 实验说明](experiments/study3_corpus_effect_v1/README.md)。三套权重均按独立 Release 分发；权重文件、词表及位置容量必须配套。本次环境与入口维护的保护快照、独立环境和三轮核验见[维护记录](reports/repository_two_studies/maintenance/README.md)。

## 目录用途

| 文件夹 | 用途 |
|---|---|
| [`src/math_tokenizer/`](src/math_tokenizer/) | 分词器、公式解析和下游模型代码。 |
| [`native/`](native/) | 加速解析、分词和解码的 C 代码。 |
| [`scripts/`](scripts/) | 运行实验、统计结果和加载模型的脚本。 |
| [`data/`](data/README.md) | 公式语料、八案例数据、MLP 缓存和模型权重。 |
| [`experiments/`](experiments/) | 实验配置、词表、结果和日志。 |
| [`reports/`](reports/) | 实验汇总表和统计图。 |
| [`eqgpt_original/`](eqgpt_original/) | EqGPT 原版代码，包含 PINN。 |
| [`参考资料/`](参考资料/) | 参考论文和参考仓库链接。 |

数据放在 `data/`：

```text
data/
├── formulas/
│   ├── pde/               # 上游公式
│   ├── eqgpt/             # EqGPT 公式表和增强后的训练公式
│   └── experiment_inputs/ # 各次上游实验使用的公式
├── cases/                 # 八个案例的原始数据
├── mlp_cache/             # 八个案例拟合好的 MLP
├── tokenizers/            # Qwen2.5、Llama 3 分词器
└── weights/
    ├── manifest.json      # 权重文件清单
    ├── pretrained/        # 预训练权重，下载后放在这里
    └── searched/          # 搜索后的权重，下载后放在这里
```

语料、原始数据和 MLP 缓存随仓库提供。方案一约 2.6 GiB 的 Transformer 权重从 [GitHub Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/eqgpt-fast-20261022-weights-v1) 下载后放入 `data/weights/`，方法见 [数据说明](data/README.md)。方案二另有36份权重，下载到其独立实验目录，见[模型使用入口](scripts/eqgpt/README_方案二模型使用.md)。

实验和结果目录：

| 子目录 | 内容 |
|---|---|
| `experiments/upstream_main/` | 方案一的三划分上游主实验。 |
| `experiments/upstream_vocab_size/` | 方案一上游词表大小实验。 |
| `experiments/upstream/` | 方案一上游日志。 |
| `experiments/upstream_811/` | 方案二的 8:1:1 上游及基础表×容量扫描。 |
| `reports/preparation_811/` | 方案二上游报告、验证汇总和准备阶段审计。 |
| `experiments/downstream/` | 方案一的原 benchmark 下游八案例实验。 |
| `experiments/expanded_downstream_80_v1/` | 扩大语料下游八案例实验（独立版本，80 token 容量）。 |
| `experiments/study3_corpus_effect_v1/` | 第三套语料规模效应实验的配置、词表绑定、逐候选结果、机制追踪和日志；`.pt` 权重由独立 Release 分发。 |
| `reports/upstream/` | 上游结果汇总。 |
| `reports/downstream/` | 下游结果汇总。 |
| `reports/expanded_downstream_80_v1/` | 扩大语料实验结果、旧实验逐案例对照和完整性核验。 |
| `reports/study3_corpus_effect_v1/` | 第三套大小语料配对结果、机制阶段统计、五轮复核和运行日志。 |
| `reports/vocab_usage/` | token 使用统计。 |

## 分词器代码与词表

Hard AST-BPE 的训练、编码和解码在 [`hard_ast_2026.py`](src/math_tokenizer/hard_ast_2026.py) 的 `HardASTBPE` 类中。训练调用 [`atoms_v4.py`](src/math_tokenizer/atoms_v4.py) 中的 `GrammarBPE`；公式结构由 [`frontend_v6.py`](src/math_tokenizer/frontend_v6.py) 解析。编码时使用 [`hard_ranked_2026.c`](native/hard_ranked_2026.c) 执行合并。

[`compact_alphabet.py`](src/math_tokenizer/eqgpt/compact_alphabet.py) 负责压缩基础字符表和映射 token ID。下游通过 [`latex_tokenizers.py`](src/math_tokenizer/eqgpt/latex_tokenizers.py) 中的 `LatexAdapter` 接入 Transformer。

训练好的词表保存在实验目录：

- **方案二上游，多基础表、64–512-ID 容量扫描**：[词表与划分](experiments/upstream_811/README.md)。
- **方案二下游，280/315/350 ID**：[实际训练词表](data/formulas/expanded_downstream_80_v1/tokenizers_round_0/)。
- **方案一上游主实验，256 ID**：[`experiments/upstream_main/`](experiments/upstream_main/)。
- **方案一上游词表大小实验，200–450 ID**：[`experiments/upstream_vocab_size/`](experiments/upstream_vocab_size/)。
- **方案一下游，280、315、350 ID**：各案例的 `tokenizers/` 目录，例如 [`Wave_equation/tokenizers/`](experiments/downstream/Wave_equation/tokenizers/)。
- **逐项查看方案一下游词表**：[280 ID](reports/downstream/vocabulary_280.md)、[315 ID](reports/downstream/vocabulary_315.md)、[350 ID](reports/downstream/vocabulary_350.md)。上游各实验目录中的 `*.tokens.csv` 还记录 token 的使用次数和频率。

每份词表需要两个文件：`ast_bpe_<size>.model.json` 保存基础字符表和 ID 数量，`ast_bpe_<size>.model.backend.json` 保存 token 字节和合并规则。加载时两个文件须放在同一目录。

方案一和两套下游使用 98 个基础字节字符；方案二上游比较多种基础表，其中可运行配置包含91、95、98字符。总词表大小包含三个控制 ID：`PAD=0`、`EOS=1`、`BOS=2`，因此 256-ID 词表有 253 个内容 token。底层 backend 还保留部分内部字节条目，其条目数不能直接作为最终词表大小。

在已安装依赖的环境中，从仓库根目录运行 Python，并设置 `PYTHONPATH=src`：

```python
from pathlib import Path
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet

path = Path("experiments/upstream_main/pde_fold_1/ast_bpe_256.model.json")
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

以下三划分结果属于方案一。方案二的 8:1:1 上游完整说明见 [upstream_811/README.md](experiments/upstream_811/README.md)，使用独立划分与词表，不与此处轮换测试结果混合。

比较 Hard AST-BPE、Qwen2.5、Llama 3 和 `o200k_base` 四个分词器，只使用 1500 条 PDE。

先按公式来源分成 3 份，每份 500 条。每次选其中 2 份作为训练集，1 份作为测试集，轮流测试，共做 3 次。为避免训练集出现与测试集结构重复的公式，第一、第三次各从训练集中删去 1 条，所以三次训练集分别有 999、1000、999 条，测试集均为 500 条。

另外做一组“用简单公式训练、用复杂公式测试”的实验，并比较 Hard AST-BPE 在 200–450 ID 下的表现。

三个指标为 **tokenizer compression、operation、numbers**，分别衡量序列压缩、LaTeX 命令完整性和数字串完整性。定义见 [上游实验说明](reports/upstream/README.md)。我们也统计 token 的使用频率和长度。

[主实验结果](reports/upstream/main_summary.csv) · [词表大小实验](reports/upstream/three_metrics.csv) · [token 使用统计](reports/upstream/per_token_usage.csv) · [日志](experiments/upstream/)

查看分词结果：

```bash
PYTHONPATH=src python scripts/当前四分词器.py hard_ast_bpe 'u_t+u_x=0'
```

其余方法名为 `qwen2.5`、`llama3` 和 `o200k_base`。实验入口为 [`evaluate_screened_pde.py`](scripts/evaluate_screened_pde.py)，词表统计入口为 [`analyze_pde_vocab_usage.py`](scripts/analyze_pde_vocab_usage.py)。

## 下游实验

以下 212 条训练语料的八案例结果属于方案一；后面的扩大语料版本属于方案二。两套均保留完整结果，不相互覆盖。

将 Hard AST-BPE（280、315、350 ID）和 EqGPT GRE（57 ID）分别接入从头训练的 GPT-2 架构模型，在八个已知 PDE 案例上比较方程结构发现结果。

这组实验缩小了原 EqGPT 的训练与搜索规模：Transformer 从 6 层改为 3 层，初始训练从 100 epoch 改为 25 epoch；搜索从 5 轮、每轮 400 次改为 3 轮、每轮 200 次；每轮继续训练从 5 epoch 改为 2 epoch。模型宽度为 768，注意力头数为 8，随机种子为 525。运行配置见各案例的 `protocol.json`，例如 [Wave 配置](experiments/downstream/Wave_equation/protocol.json)。

同一案例的各模型共用已拟合的 MLP surrogate。搜索保留原最小二乘 reward、候选排序和 Top-10 继续训练，不运行末尾的 PINN 系数拟合；本轮评价方程结构的恢复情况。训练语料的目标公式排除记录见 [审计文件](reports/downstream/target_exclusion_audit.json)。

[结果汇总](reports/downstream/对比结果.csv) · [逐案例结果和日志](experiments/downstream/) · [运行脚本](scripts/eqgpt/fast_reward_case.py)

已训练的 36 份 Transformer 权重在 [GitHub Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/eqgpt-fast-20261022-weights-v1)。使用方法见 [模型加载说明](scripts/eqgpt/README_权重.md)，无需重新训练已有案例的 MLP。

### 扩大语料版本（独立实验）

扩大语料实验单独保存在 [`experiments/expanded_downstream_80_v1/`](experiments/expanded_downstream_80_v1/)，不会覆盖上面的旧实验。两套实验使用相同八个已知 PDE 案例、四个模型 arm、25 epoch、种子 525、3 轮×200 次搜索及每轮 2 epoch continuation；新版本将训练/搜索容量从 50 提高到 80，并以 1,855 条共同训练源（1,643 条新外部 benchmark + 原有 212 条非目标 EqGPT 公式）替代旧实验每个目标排除后使用的 212 条公式。新旧语料、增广方法、Hard 输入表示和搜索解析路径也有变化，详细差异见[对照报告](reports/expanded_downstream_80_v1/comparison_previous_zh.md)。因此两轮结果不能解释为只增加语料量的单变量实验。

两个 benchmark 是不同来源的公式集：旧 benchmark 是 EqGPT 手册中的 221 条 PDE 公式；新外部 benchmark 是经来源和结构审查的独立公式集，容量筛选前 2,045 条，80 容量共同筛选后实际训练使用其中 1,643 条。新语料另外保留旧 benchmark 排除八个测试目标后的 212 条公式，故本次总训练池为 1,855 条。两个 benchmark 共用 GRE 结构表示和八案例测试集，但来源、规模和用途不同；旧目录和旧结果均保留。组成、排除与来源见[数据说明](data/README.md)和[新外部 benchmark 说明](data/formulas/gre_benchmark_filtered/README.md)。

新版本的逐案例候选、训练记录、配置、词表和结构化运行记录见实验目录；每个模型 arm 均有 `fast_summary.json`，完整结构化结果见[完整结果 JSON](reports/expanded_downstream_80_v1/完整结果.json)。主要结论见[实验报告](reports/expanded_downstream_80_v1/report_zh.md)，新旧逐项比较见[对照 CSV](reports/expanded_downstream_80_v1/comparison_previous.csv)。36 份训练后 Transformer 权重约 2.78 GB，和旧实验相同，通过[独立 GitHub Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/expanded-downstream-80-v1-weights)分发，不放入 Git 历史；每个权重的 SHA-256 和大小见 [`weight_manifest.json`](experiments/expanded_downstream_80_v1/weight_manifest.json)，下载和校验命令见[实验 README](experiments/expanded_downstream_80_v1/README.md)。

## 数据与参考资料

[上游公式集](data/formulas/pde/) 包含 1500 条 PDE，来自 490 个来源组，简单、中等、复杂各有 450、600、450 条。目录还保留 55,644 条非 PDE 数学公式，当前上游实验未使用这些公式。筛查记录包括来源、自动检查和语义抽查，尚未逐条经过专家审核。

下游八个测试案例的原始数据在 [`data/cases/`](data/cases/)，已拟合 MLP 在 `data/mlp_cache/<case>/surrogate_cache.pt`。具体路径见 [数据索引](data/README.md)。这些案例用于已知方程的结构恢复，仓库尚未包含未知控制方程的发现实验。

[参考资料](参考资料/README.md) 收录论文 PDF 和参考仓库链接。[EqGPT 论文原版说明](eqgpt_original/论文原版说明.md) 对应作者的完整代码，与本项目的缩小版实验分开保存。

环境安装与历史依赖区别见 [ENVIRONMENT.md](ENVIRONMENT.md)，当前环境直接依赖参考见 [requirements-runtime.lock.txt](requirements-runtime.lock.txt)。原有两份依赖文件保留其历史身份，不应同时安装来覆盖版本。项目配置见 [`pyproject.toml`](pyproject.toml)。许可与第三方材料说明见 [`LICENSE`](LICENSE) 和 [`NOTICE.md`](NOTICE.md)。
