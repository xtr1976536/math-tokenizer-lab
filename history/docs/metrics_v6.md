# v6 分词器本征评估：来源、定义与使用限制

本轮只评估分词器，不训练下游语言模型。主指标为按论文等权的 token/UTF-8 byte，另报告整库平均 token/公式和长度尾部。压缩与输入长度相关的几种表示不是几份独立证据，综合分只使用其中一项。

文献检索使用论文官网与作者代码。搜索工具本轮持续返回 HTTP 502/503，因此通过 HTTPS 直接取得 ACL Anthology 页面、PDF 和官方 BibTeX；保存论文哈希及完整元数据，见 `references/metrics_v6_sources.json` 和 `references/metrics_v6.bib`。

| 指标组 | 本轮指标 | 依据与含义 | 是否进入综合排名 |
|---|---|---|---|
| 分词长度 | paper-macro token/byte，token/公式，bytes/token，P50/P95/P99 | TokEval；长度与实际上下文占用有关；多个单位彼此依赖 | 仅 paper-macro token/byte |
| 可逆性 | 原字符串 exact roundtrip、独立 token payload 重构 | TokEval reconstruction fidelity；保持空白、Unicode 和原始 LaTeX | 全输入无损是排名资格门槛 |
| 分布与利用 | vocabulary utilization；Rényi efficiency α=1,2,2.5,3；empirical unigram bits/byte | Zouhar 等 2023；TokEval 官方函数另做复算 | 描述性，不能用作效果保证 |
| 泛化编码代价 | fit-predictive unigram bits/byte，fit-rare token exposure | 训练集 unigram 概率加 0.5 平滑，独立测试平均负对数概率，按原始 bytes 归一 | predictive BPB 进入；不是 LM perplexity |
| 数学词法 | command integrity、number integrity、lexeme preservation、operator isolation、boundary precision/recall/F1 | TokEval 数字、算子、UTF-8 与边界指标的数学版本；不使用空白 word fertility 冒充公式词法 | command 与 number 进入；其余辅助 |
| 数学结构 | 独立 KaTeX syntax AST source-span crossing；可测比例 | 领域扩展，由固定 KaTeX 解析，所有方法同一份参考；不是已获普遍认可的数学语义金标准 | crossing 进入，但明确局限 |
| 表面变化 | 受限布局变化下 content-token 数变化 | 领域鲁棒性探针，跳过文字/命名算子内容的变换；不宣称任意语义等价 | 辅助 |
| 工程效率 | raw string→IDs throughput、IDs→string throughput、训练时间、磁盘体积、加载时间 | TokEval speed 及公开接口；包括我们原生前端与 RL 原子 codec 成本 | 编码速度进入 |
| 泛化分层 | 新论文、完全排除目标学科的训练→目标学科测试；领域、来源、复杂度及长公式分层 | paper-disjoint/category-disjoint 设计；来源是研究变量 | 各场景单列，综合场景各 50% |

[Tokenization Is More Than Compression (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.40/) 给出了对“最短分词必然更好”的直接反证，也提供公开 PathPiece 实现。因此本轮使用未经改动的作者 Rust 代码作为独立算法基准，使用相同 fit 集训练的 BPE 词表，实际 ID 预算完全一致。

[Tokenization and the Noiseless Channel (ACL 2023)](https://aclanthology.org/2023.acl-long.284/) 提出 Rényi efficiency。[Two Counterexamples to Tokenization and the Noiseless Channel (LREC-COLING 2024)](https://aclanthology.org/2024.lrec-main.1469/) 展示该指标可提高而下游效果下降。这里保留信息分布读数，但不把它作为单一优胜标准，也不据此推断下游优势。

[Subword Regularization (ACL 2018)](https://aclanthology.org/P18-1007/) 的 UnigramLM 通过官方 SentencePiece 实现；[BPE-Dropout (ACL 2020)](https://aclanthology.org/2020.acl-main.170/) 是训练时随机分词方案，确定性推理设置 dropout=0 后与相同 BPE 重合，故不重复记作独立胜负证据。[VOLT (ACL 2021)](https://aclanthology.org/2021.acl-long.571/) 主要研究词表规模/粒度选择；本轮不将我们写的近似算法冒充官方 VOLT 实验。

用户提供的 Dauncey 2026 论文主要启发 RL 中相对奖励和降低方差。其方法把 token boundary policy 与语言模型预测联合训练；本轮独立词表的 terminal RLOO 不能称为该论文的复现。我们保持训练阶段学习离散合并动作、上线部署固定词表的路线；词法原子、奖励权重、词表预算仍有设计者选择，不能声称彻底无需人工介入。离线训练指部署前训练，不指 logged-data offline RL。

综合排名权重在首次测试评分前冻结：压缩 30%、编码速度 25%、独立结构 crossing 20%、命令完整性 10%、数字完整性 5%、fit-predictive BPB 10%。使用同词表预算、同数据范围的方法百分位排名；固定预训练大词表参考单列。权重没有普适科学定理保证，因此还提供单指标排序、Pareto 集和权重敏感性。熵、词表利用率、长度的重复单位不额外加分。
