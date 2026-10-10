# 第二版已更新（2026-10-11）

最新语料和实验请使用 **v2**：28,347条公式（首版的16.55倍），98篇固定在2020年前的arXiv源稿，historical human 19,675条，simple/complex 24,109/4,238。13个主题领域；每条保留来源/许可/版本/位置/复杂度分量。所有arXiv版本经CC BY/CC0核对，缓存源码不上传。

- 当前数据：`data/releases/v2/curated/corpus.jsonl`；纯公式：`data/releases/v2/formula_only/<domain>/<origin>/<simple|complex>.jsonl`。
- 完整协议、合法获取、复杂度规则：`docs/arxiv_collection_v2.md`。
- 独立评测指标与文献：`docs/intrinsic_metrics_v2.md`；RL设计和全部消融：`docs/rl_design_v2.md`。
- 跨预算结果：`docs/v2_findings.md`；正式实验：`results/v2_release512/`和`results/v2_release1024/`。
- 复现：`docs/v2_reproduction.md`、`scripts/run_v2_all.sh`、`requirements-v2.lock.txt`。
- 旧v1数据、模型、结果保留，见下面首版记录；不要把v1数字当最新结果。

v2结构前端为语义AST + 明确标注的LaTeX syntax tree回退，不冒称通用语义AST。RL-v2用BPE固定词表上的完整路径REINFORCE策略，实验相对奖励基线、RLOO、折扣、熵、采样数、初始化与训练轮次；每预算38个dev配置/种子run，配置按dev冻结后才统一test。

以下为首版历史记录：

# Math Tokenizer Lab

专门用于数学公式的独立分词器研究。**当前交付是可运行、可审计的第一阶段 pilot，不是整个项目完成或下游性能提升的证明。**

- 1,713 条纯 LaTeX 公式；PDE + 九类非 PDE。原始数学结构人工来源 229 条、可能混合 AI 1,287 条、助手生成模板 51 条、证据不足 146 条。
- `data/formula_only/` 每行只含一个 LaTeX JSON 字符串；`data/curated/` 是带来源和标签的记录；`data/quarantine/` 是拒收审查材料，绝不输入模型。
- AST-BPE、RL-BPE、RL-AST-BPE 三种独立实现；完全 UTF-8 回退，无 GPT-2 或其他下游模型。
- 384 / 512 两个预算，各三个自研随机种子；每个设置 26 个方法/种子记录；另有 14 个小样本 smoke 记录。
- 原版 HuggingFace BPE、SentencePiece BPE/Unigram、tiktoken GPT-2/cl100k/o200k、Qwen2.5 官方 tokenizer 基准，第三方代码未修改。
- 53 项自动测试、六轮独立工程检查、四轮公平性审计；模型、公式逐条结果、版本、配置及哈希均在仓库。

## 先看哪些文件

| 内容 | 路径 |
|---|---|
| 数据统计、规范哈希 | `data/curated/summary.json` |
| 纯公式 | `data/formula_only/by_domain/<领域>/<来源>.jsonl` |
| 全部带证据记录 | `data/curated/corpus.jsonl` |
| 固定来源／许可证／文件哈希 | `data/source_registry.json` |
| 研究协议、来源定义与方法区别 | `docs/research_protocol.md` |
| 384 预算完整结果 | `results/pilot01/report.md` |
| 512 预算敏感性结果 | `results/sensitivity512/report.md` |
| 研究结果解释与下一步 | `docs/findings.md` |
| 六轮检查 | `results/pilot01/six_round_checks.json` |
| 四轮公平性审计 | `results/pilot01/fairness_audit.json` |
| 阶段完成／未完成清单 | `docs/status.json` |

“human”是原数学结构来源证据，LaTeX 是助手写代码后机械转换，另有 `latex_author_origin`。不能将转换后的每条 LaTeX 宣称人工敲入。非 PDE 的严格人工桶尚为空，unknown 不能并进去。

EqGPT 数据固定为 `dc35fc0ac860ff224293f01d2a62204042095127`：工作簿版本及多个工作表不同于论文报告的 221 条，因此当前记录 229 条去重结构，保留原串及行号。它们缺系数与符号，是 coefficient-free PDE 结构表达式，未伪造完整等式。

OpenAI 最近公开的数学稿件已使用，固定 `fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb`，按官方人工编辑例外信息保守记 possible_mixed。公式数学真伪未由本项目证明。

## 结果概要

共同测试集 406 条；所有已运行方法 100% 精确还原。自研与同训练预算基准如下（自研列为三个种子均值；token/公式越小越短）：

| 方法 | 384 词表预算 | 512 词表预算 |
|---|---:|---:|
| byte BPE | 14.377 | 12.616 |
| AST-BPE | 17.537 | 16.818 |
| RL-BPE | 14.479 | 12.532 |
| RL-AST-BPE | 17.644 | 16.843 |
| HF byte BPE | 14.377 | 12.616 |
| SentencePiece BPE | 17.631 | 14.357 |
| SentencePiece Unigram | 17.606 | 14.204 |

AST 测试覆盖率仅 198/406=48.8%。结构方法在可解析子集的跨子树 token 比例为零，但全测试压缩受字节回退影响。512 预算在可解析子集及小样本 PDE 子群有局部优势，不得外推为整体优势。RL-BPE 在 512 平均略缩短、384 略变长，现阶段不支持稳定优于 BPE 的结论。组 bootstrap 区间见完整报告。

成熟固定词表（GPT-2/Qwen/OpenAI）使用相同 raw-string 指标，单列 pretrained_reference；词表和外部语料不同，不能与 384/512 训练组作公平训练算法排名。

## 本地安装与顺序复现

```bash
# 从本仓库根目录执行，需要 Python 3.11+ 和 uv
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.lock.txt

# 仓库已带公式数据、模型与结果：先运行检查，无需下载全文
.venv/bin/python -m pytest tests -q
.venv/bin/python -m math_tokenizer.audit results/pilot01
.venv/bin/python -m math_tokenizer.audit results/sensitivity512

# 需要重建原始提取时：逐文件下载固定快照，验证 SHA-256
.venv/bin/python scripts/fetch_sources.py
.venv/bin/python scripts/fetch_baselines.py
.venv/bin/python -c 'from math_tokenizer.corpus import build; print(build())'
.venv/bin/python scripts/export_formula_only.py

# 新输出目录：完成的结果不能覆盖。下列命令顺序执行。
.venv/bin/python -m math_tokenizer.run --output results/reproduce384 --capacity 384 --seeds 17 29 43
.venv/bin/python -m math_tokenizer.run --output results/reproduce512 --capacity 512 --seeds 17 29 43
.venv/bin/python -m math_tokenizer.audit results/reproduce384
.venv/bin/python -m math_tokenizer.audit results/reproduce512

# 六轮原运行审查含原始来源哈希、保存模型重放及优化前后对照
.venv/bin/python scripts/checks.py --result results/pilot01
```

也可运行 `bash scripts/run_all.sh <新的输出目录前缀>`。正常使用只需拉取公式、源码和模型；全文原始素材和第三方框架不在主仓库。tiktoken 的官方 encoding 文件首次使用会下载缓存。

保存模型可直接独立用：

```python
from pathlib import Path
from math_tokenizer.bpe import MathBPE
model = MathBPE.load(Path('results/pilot01/models/ast_bpe-17.json'))
text = r'\frac{\partial u}{\partial t}=u^{2}'
ids = model.encode(text)
assert model.decode(ids) == text
```

## 与第二篇论文的关系

[Dauncey & Wattenhofer, ICML 2026](https://arxiv.org/abs/2602.13940) 是端到端语言模型内部的动态字节边界学习，不是静态 BPE 合并表训练。这里借用 score-function/REINFORCE 思想，采用独立压缩奖励。不能称作其原算法复现。官方代码原样克隆、commit 记录在 `references/upstream_code.json`；CUDA/Triton 动态方法未在本机运行，未加入已跑通基准表。

## 尚未完成

严格人工非 PDE LaTeX 的补充与专家逐条审查；通用语义 AST；RL 长期收益／语义目标与系统消融；新论文 standalone 基准；arXiv 小批量版本化提取；下游提升的后续验证。现在只完成支持这些工作继续开展的第一阶段，详见 `docs/status.json`。需要修复数据时发布新版本，保留当前输入与结果。

## 来源与许可证

[EqGPT](https://github.com/woshixuhao/EqGPT)、[OpenAI 数学稿件](https://github.com/openai/math)、[SymPy 1.10](https://github.com/sympy/sympy/tree/sympy-1.10)、[Dauncey 原代码](https://github.com/SamD770/bitter-lesson-tokenization)。代码 MIT，公式来源声明继承相应上游许可；许可证快照和声明见 `references/` 与 `NOTICE.md`。助手合成模板明确 CC0，不扩大上游授权范围。
