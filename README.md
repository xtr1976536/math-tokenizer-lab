# 数学公式分词与 PDE 方程发现

本仓库有四块内容：**上游任务、下游任务、benchmark、参考资料**。仓库现已公开，说明文档目前使用中文。本文中的 equation-BPE 指本项目使用的 **Hard AST-BPE**。

## 项目结构与目录用途

下面列出 GitHub 中保存的主要目录。上游任务评估分词器，下游任务评估 PDE 方程结构发现。

```text
math-tokenizer-lab/
├── src/math_tokenizer/                 # 分词器、数学公式解析和评估实现
│   ├── hard_ast_2026.py                # equation-BPE / Hard AST-BPE 核心算法
│   └── eqgpt/                         # 下游接入、训练与方程搜索
│       ├── compact_alphabet.py        # 压缩基础字符表、内容 ID 映射与词表读写
│       ├── latex_tokenizers.py        # 连续 LaTeX 序列适配及控制 token
│       └── eqgpt_fast_tokenizers.py   # 下游固定词表预算的训练与加载
├── native/                            # 数学公式结构解析、分词及解码的 C 实现
├── scripts/                           # 实验、统计与已训练模型的使用入口
│   └── eqgpt/                         # 缩小版下游实验与模型权重加载入口
├── models/                            # Qwen2.5、Llama 3 的通用分词器文件及来源
├── data/benchmark_pde1500_20261022/    # 上游公式、来源和固定划分
├── benchmark/                         # 下游八案例数据、MLP 缓存和权重索引
├── experiments/
│   ├── pde1500_main_final_20261022/    # 上游主实验：逐公式结果及冻结词表
│   ├── pde1500_sweep_200_450_20261022/ # 上游词表规模扫描：结果及冻结词表
│   ├── upstream/                      # 上游运行日志
│   └── downstream/                    # 八个测试案例：词表、结果、日志及 MLP 缓存
├── reports/                           # 实验汇总、词表清单、token 频率和统计图
│   ├── upstream/                      # 四种分词器的上游结果
│   ├── downstream/                    # Hard AST-BPE 与EqGPT 的下游结果
│   └── pde1500_vocab_200_450_20261022/ # 词表规模与 token 使用分布分析
├── third_party/eqgpt/                 # EqGPT 论文原版代码及八个测试案例的原始数据
├── provenance/upstream_original/      # 上游原始实验来源与记录
└── 参考资料/                           # 参考论文 PDF、原仓库链接与许可说明
```

根目录的 [`pyproject.toml`](pyproject.toml)、[`requirements.lock.txt`](requirements.lock.txt) 和 [`requirements-eqgpt.lock.txt`](requirements-eqgpt.lock.txt) 保存包配置与实验依赖；[`LICENSE`](LICENSE) 和 [`NOTICE.md`](NOTICE.md) 说明本项目许可及第三方材料来源。较大的 Transformer 权重存放于 [GitHub Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/eqgpt-fast-20261022-weights-v1)，文件索引见 [`benchmark/weights_manifest.json`](benchmark/weights_manifest.json)。`models/` 中保存的是通用分词器文件，不是这些下游 Transformer 权重。

## equation-BPE 在哪里定义？

| 功能 | 实现位置 | 主要入口 |
|---|---|---|
| Hard AST-BPE 的训练、编码、解码与模型保存 | [`src/math_tokenizer/hard_ast_2026.py`](src/math_tokenizer/hard_ast_2026.py) | `HardASTBPE.train / encode / decode / save / load` |
| 训练时的合并统计与 AST 边界约束 | [`src/math_tokenizer/atoms_v4.py`](src/math_tokenizer/atoms_v4.py)；[`src/math_tokenizer/frontend_v6.py`](src/math_tokenizer/frontend_v6.py) | `GrammarBPE`；`structure_native` |
| 按训练得到的合并顺序执行分词 | [`native/hard_ranked_2026.c`](native/hard_ranked_2026.c) | 由 `HardASTBPE.encode` 调用；Python 参照实现为 `encode_reference` |
| 压缩基础字符表及最终内容 ID 映射 | [`src/math_tokenizer/eqgpt/compact_alphabet.py`](src/math_tokenizer/eqgpt/compact_alphabet.py) | `CompactAlphabet`；`train_compact` |
| 上游四分词器的加载与分词 | [`scripts/当前四分词器.py`](scripts/当前四分词器.py) | `load`；默认读取第一折的 256-ID 词表 |
| 下游词表训练及连续 LaTeX 接入 | [`src/math_tokenizer/eqgpt/eqgpt_fast_tokenizers.py`](src/math_tokenizer/eqgpt/eqgpt_fast_tokenizers.py)；[`src/math_tokenizer/eqgpt/latex_tokenizers.py`](src/math_tokenizer/eqgpt/latex_tokenizers.py) | `train_or_load`；`LatexAdapter` |

## 训练好的 equation-BPE 词表保存在哪里？

**每份冻结词表由两个同目录的 JSON 文件组成，加载时必须同时保留。** `ast_bpe_<size>.model.json` 记录基础字符表、最终内容 ID 数和配套文件名；`ast_bpe_<size>.model.backend.json` 保存 `vocab_hex`（token 字节的十六进制表示）与 `merges`（有序合并规则）。`<size>` 是包含三个控制 ID 的总词表大小。

| 实验 | 保存位置与直接入口 |
|---|---|
| 上游主实验，256 ID | `experiments/pde1500_main_final_20261022/<split>/`；示例：[第一折模型配置](experiments/pde1500_main_final_20261022/pde_fold_1/ast_bpe_256.model.json)、[配套词表与合并规则](experiments/pde1500_main_final_20261022/pde_fold_1/ast_bpe_256.model.backend.json)、[可直接查看的 token 与频率 CSV](experiments/pde1500_main_final_20261022/pde_fold_1/ast_bpe_256.tokens.csv) |
| 上游 200–450 ID 扫描 | [`experiments/pde1500_sweep_200_450_20261022/`](experiments/pde1500_sweep_200_450_20261022/)，按 `<split>/ast_bpe_<size>.model.json` 保存，同目录包含配套 backend 和 token CSV |
| 下游 280、315、350 ID | `experiments/downstream/<case>/tokenizers/`；示例：[Wave 280](experiments/downstream/Wave_equation/tokenizers/ast_bpe_280.model.json)、[Wave 315](experiments/downstream/Wave_equation/tokenizers/ast_bpe_315.model.json)、[Wave 350](experiments/downstream/Wave_equation/tokenizers/ast_bpe_350.model.json)，各有同名配套 backend |
| 下游逐 ID 的可读词表 | [280-ID 清单](reports/downstream/vocabulary_280.md)、[315-ID 清单](reports/downstream/vocabulary_315.md)、[350-ID 清单](reports/downstream/vocabulary_350.md) |

上游 `<split>` 为 `pde_fold_1`、`pde_fold_2`、`pde_fold_3` 或 `pde_simple_to_complex`；下游 `<case>` 的八个名称见 [benchmark 索引](benchmark/README.md)。不同划分或案例应读取各自冻结的词表，不能只按词表大小互换模型文件。

当前已保存实验的基础表为 **98 个字节字符**。例如 256-ID 词表包含 253 个内容 token 和 3 个控制 ID。backend 内部保留完整 256 字节作为算法实现的基础结构，**其 `vocab_hex` 条目数不等于压缩后的最终词表大小**；最终有效词表由 `CompactAlphabet` 过滤和映射得到。`model.vocab` 与 `model.encode` 使用从 0 开始的内容 ID；下游 `LatexAdapter` 将内容 ID 加 3，并使用 `0=<PAD>`、`1=<EOS>`、`2=<BOS>`。

在仓库根目录、已安装项目依赖的 Python 环境中，可以加载并逐项查看实际词表：

```python
from pathlib import Path
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet

path = Path("experiments/pde1500_main_final_20261022/pde_fold_1/ast_bpe_256.model.json")
model = CompactAlphabet.load(path)  # 自动读取同目录的配套 backend JSON
for token_id, token in enumerate(("<PAD>", "<EOS>", "<BOS>")):
    print(token_id, token)
for content_id, token_bytes in enumerate(model.vocab):
    print(content_id + 3, repr(token_bytes.decode("utf-8", errors="backslashreplace")))

formula = "u_t+u_x=0"
content_ids = model.encode(formula)
print("内容 ID:", content_ids)
print("还原公式:", model.decode(content_ids))
assert model.decode(content_ids) == formula
```

运行上述示例前可设置 `PYTHONPATH=src`。首次加载会自动编译 C 后端，需要可用的 C 编译器。输入包含基础字符表以外的字节时，会明确报错；模型不会静默替换为未知 token。

## 1. 上游任务

我们评估 **Hard AST-BPE、Qwen2.5、Llama 3、`o200k_base`** 四个分词器。最新主实验使用 1500 条 PDE 做三折测试和“简单训练→复杂测试”，并扫描 Hard AST-BPE 的 200–450 总词表规模。三个指标统一命名为 **tokenizer compression、operation、numbers**，分别对应旧报告的压缩率、Cmd、Num；只改名称，不重算数据。另保留 token 使用次数及概率。

- [主实验结果](reports/upstream/main_summary.csv)、[词表规模指标](reports/upstream/three_metrics.csv)、[逐 token 使用统计](reports/upstream/per_token_usage.csv)
- [主实验逐公式记录与冻结词表](experiments/pde1500_main_final_20261022/)、[Hard 词表扫描](experiments/pde1500_sweep_200_450_20261022/)
- [主实验日志](experiments/upstream/pde1500_main_final_20261022.log)、[扫描日志](experiments/upstream/pde1500_sweep_200_450_20261022.log)（均从同批原始日志提取本仓库当前方法的行）
- [四分词器加载脚本](scripts/当前四分词器.py)、[通用词表文件](models/)、[Hard 实现](src/math_tokenizer/hard_ast_2026.py)

调用示例：`PYTHONPATH=src python scripts/当前四分词器.py hard_ast_bpe 'u_t+u_x=0'`。其余方法名为 `qwen2.5`、`llama3` 和 `o200k_base`。

重新评估固定公式集可运行 `PYTHONPATH=src:. python scripts/evaluate_screened_pde.py --output experiments/复核主实验 --sizes 256`。

## 2. 下游任务

比较  AST-BPE（280、315、350 ID）与 **EqGPT GRE（57 ID）** 接入同一个gpt-2架构的llm，并在八个已知 PDE 案例中搜索正确方程结构。两个模型均从头训练；每个案例共用一个预先拟合并缓存的 MLP surrogate，保留 EqGPT 原最小二乘 reward、候选排序和 Top-10 继续训练。末尾 PINN 系数拟合没有运行，因为我们关注的是**公式结构是否可以被发现**，而不是系数精度。

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

实际参数以[保存的运行协议](experiments/downstream/Wave_equation/protocol.json)为准；这是**缩小后的对照实验**，不能等同于原论文的完整预算。[Hard 与 GRE 对比结果](reports/downstream/对比结果.csv)、[八案例完整搜索记录](experiments/downstream/)和[目标公式排除审计](reports/downstream/target_exclusion_audit.json)均已保存。每个案例的运行日志从同批原始日志中提取了共享 surrogate、Hard 和 GRE 的行；结果文件只收录 Hard 和 GRE。已训练的 36 个 Transformer 权重在[模型权重 Release](https://github.com/xtr1976536/math-tokenizer-lab/releases/tag/eqgpt-fast-20261022-weights-v1)，[下载与直接使用说明](scripts/eqgpt/README_权重.md)提供生成、评价和继续搜索示例。

## 3. Benchmark

- [上游固定公式集](data/benchmark_pde1500_20261022/)：我们从 490 篇论文中筛选 1500 条 PDE 公式，并按语法结构分为简单/中等/复杂，比例为 450/600/450；此外保留 55,644 条非 PDE 数学公式。质量检查包括来源回放、解析检查和语义抽查，**没有对全部 1500 条逐条进行专家人工检查**。
- [下游八案例数据与已拟合 MLP 缓存索引](benchmark/README.md)、[EqGPT 原作者 PDE 结构工作簿](third_party/eqgpt/code/PDE_dataset_0725.xlsx)：这八案例是**已知方程的测试**，目标公式未进入模型的训练语料；我们尚未在前人也未知其有效控制方程的新数据集上完成发现实验。

## 4. 参考资料

[参考论文 PDF、原仓库地址与许可说明](参考资料/README.md)。[EqGPT 论文原版完整代码](third_party/eqgpt/论文原版说明.md)单独保存，包含原作者的 PINN 全流程，和本仓库缩小版运行代码分开。第三方原始源码保持原貌；仓库说明先使用中文，待内容确认后再制作英文版。
