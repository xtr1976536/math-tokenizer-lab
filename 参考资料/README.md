# 参考论文与仓库

## PDF

- [AST-BPE 原始论文](PDF/AST-BPE_原论文.pdf)：Müller 与 Vlachos，*Byte-Pair Encoding for Text-to-SQL Generation*，arXiv:1910.08962。数学公式上的 hard AST-BPE 是本项目对“按结构限制合并”原则的改造，并非原论文提供的数学分词器。
- [EqGPT 原论文](PDF/EqGPT_原论文.pdf)：Xu 等，*Generative discovery of partial differential equations by learning from math handbooks*，*Nature Communications* 16, 10255 (2025)。本仓库的下游实验是缩小后的对照实验，不复现原论文的完整计算预算。

这些 PDF 是阅读资料。论文的著作权归各自作者及出版方；重新分发、引用或改编时应遵守各自许可。

## 原仓库与工具

| 项目 | 链接 | 本仓库用途 |
|---|---|---|
| EqGPT 原作者仓库 | https://github.com/woshixuhao/EqGPT | 下游实验原作者代码及八案例数据来源；本地副本见 `eqgpt_original/` |
| Qwen2.5-0.5B | https://huggingface.co/Qwen/Qwen2.5-0.5B | 上游固定通用分词器 |
| Llama 3 | https://github.com/meta-llama/llama3 | 上游固定通用分词器；原作者 tokenizer 文件与许可证见 `data/tokenizers/Meta-Llama-3-8B/` |
| tiktoken | https://github.com/openai/tiktoken | `o200k_base` 的官方实现，由安装依赖提供 |
| Hugging Face tokenizers | https://github.com/huggingface/tokenizers | Qwen 词表的加载实现 |

版本、文件校验值及第三方许可见各数据目录、`eqgpt_original/provenance.json` 和 `data/tokenizers/llama3_provenance.json`。
