# 许可与来源说明

根目录 `LICENSE` 仅适用于本项目原创代码，不改变第三方源码、分词器词表、论文或公式数据的许可证。

- EqGPT 原作者源码、八案例数据及 PDE 公式表来自 [EqGPT 仓库](https://github.com/woshixuhao/EqGPT)；原许可和文件哈希见 `eqgpt_original/LICENSE`、`eqgpt_original/provenance.json`。
- Qwen 分词器文件与 Llama 3 分词器文件分别遵守各自原项目许可；Llama 许可证见 `data/tokenizers/Meta-Llama-3-8B/LICENSE`，来源校验见 `data/tokenizers/llama3_provenance.json`。`o200k_base` 由安装的 tiktoken 提供。
- 1500 条 PDE 及非 PDE 公式保留逐记录来源与许可信息。公式筛查和分类用于本项目实验，不代表原论文作者认可或专家逐条验证。
- 新外部 PDE benchmark 的公式、参数族实例和来源摘录见 `data/formulas/gre_external/`、`data/formulas/gre_benchmark_filtered/` 与 `data/formulas/expanded_downstream_80_v1/`。来源记录保留论文链接、版本、位置和哈希；原论文、数据库和数据源各自的许可仍适用，引用或再利用时应核对并遵循对应来源条款。`gre_external/sources/` 收录了用于核验的公开页面、方程摘录和一份来自 NSF repository 的作者稿；这些材料的版权仍归各自权利人，仓库收录不改变其许可或再分发条件。
- `参考资料/PDF/` 中论文的版权归相应作者及出版方；引用、分享或改编应遵守各自许可。
