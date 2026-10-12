# 新一轮结果：离线RL词表与真正的分词泛化

新语料53,858条，其中新增25,511条；新增119篇pre2020 CC BY/CC0源码、117篇有通过筛选公式。旧v2数据完整保留。13领域、simple/complex=45,123/8,735。
新paper test4375条/30篇，PDE/ODE/analysis专用模型→未见目标域test1636条/14篇，机械stress693条单列。test哈希先冻结，两场景dev选择完成才解锁test，没有按新test成绩调整方法。
RL重新学习词表合并，不学习在线边界；导出标准Rust BPE，推断不运行policy/AST parser，无手工命令清单。reward含code cost、词法边界、rare暴露、空白变体一致性，length-only单列。

| 场景 | 方法 | token/公式 | 命令完整率 | 跨树比例 | 编码MiB/s |
|---|---|---:|---:|---:|---:|
| broad/fresh_paper_test | rl_composite | 21.7221 | 87.75% | 26.22% | 8.684 |
| broad/fresh_paper_test | rl_length_only | 22.7858 | 64.32% | 28.26% | 8.728 |
| broad/fresh_paper_test | frequency | 22.1778 | 78.26% | 25.72% | 9.004 |
| broad/fresh_paper_test | reward_greedy | 21.5458 | 89.89% | 25.68% | 8.856 |
| broad/fresh_paper_test | hf_bpe_fulltrain | 22.3751 | 76.26% | 25.92% | 8.644 |
| broad/fresh_paper_test | hf_bpe_matched_fit | 22.1778 | 78.26% | 25.72% | 8.590 |
| broad/fresh_paper_test | sp_bpe | 27.9790 | 0.00% | 25.23% | 11.208 |
| broad/fresh_paper_test | sp_unigram | 29.0197 | 0.00% | 19.14% | 22.090 |
| specialized/extreme_domain_test | rl_composite | 15.5892 | 71.86% | 15.03% | 7.112 |
| specialized/extreme_domain_test | rl_length_only | 15.7274 | 52.24% | 17.05% | 7.091 |
| specialized/extreme_domain_test | frequency | 16.0043 | 47.68% | 18.99% | 7.170 |
| specialized/extreme_domain_test | reward_greedy | 15.5012 | 74.69% | 15.38% | 7.361 |
| specialized/extreme_domain_test | hf_bpe_fulltrain | 15.7139 | 58.63% | 16.20% | 7.387 |
| specialized/extreme_domain_test | hf_bpe_matched_fit | 16.0043 | 47.68% | 18.99% | 7.339 |
| specialized/extreme_domain_test | sp_bpe | 17.0795 | 0.00% | 21.32% | 10.792 |
| specialized/extreme_domain_test | sp_unigram | 16.7604 | 0.00% | 18.47% | 17.971 |

## 哪些结论成立

1. 速度路线有效：离线策略学merge table后，在线直接官方Rust BPE。新paper RL8.68MiB/s、HF全train8.64；未见领域RL7.11、HF7.39。相同public per-formula接口、每轮清官方BPE缓存（若API支持）、5次中位，达到普通BPE速度量级。SentencePiece仍更快，不能声称达到所有成熟tokenizer速度。
2. 新paper泛化改善：RL21.722对HF22.375，约2.92%缩短；command87.75%对76.26%。文档bootstrap delta=-.653，95%[-1.021,-.257]，当前新paper样本支持压缩优势。
3. 复合奖励优于纯长度奖励的一些核心指标：新paper21.722对22.786，command87.75%对64.32%；说明只奖励短序列不足。对应统计区间见pairwise，不外推到下游数学表现。
4. 真正未见领域的结果更保守：specialized未用algebra/geometry/number_theory/combinatorics/logic及其cross-list类别，RL15.589对HF15.714，约.79%缩短，但14篇文档CI[-.334,.100]跨0，不能称显著压缩优势。command71.86%对58.63%，tree15.03%对16.20%，这些intrinsic代理有改善。
5. 数学语料基准初版质量检查通过：来源许可/version/署名/位置、公式filter、exact去重、train-test syntax-template去重及paper隔离。专家语义/标签审核未完成，故明确是research candidate benchmark，不是gold semantic benchmark。

## 哪些结论不成立

1. RL没有稳定超越同复合reward的直接贪心控制。新papergreedy21.546、command89.89%，RL21.722、87.75%；目标域greedy15.501/74.69%，RL15.589/71.86%。差值CI有跨0，但不能反过来宣称RL最佳。当前有限候选收益已全观测，greedy天然强，真正长期规划尚未实现。
2. 没有所有指标优秀：新paperRLtree26.22%，HF25.92%，略差；Rényi2也略低。撤掉hard AST会牺牲先前zero-crossing与近99.5%命令完整性。更少人工约束不自动保持全部结构指标，v4严格版仍有用途。
3. 极端stress没有统一压缩优势。specialized aggregateRL109.297 token/formula vsHF108.036（约1.17%更长），command88.53%对83.34%，存在压缩/边界取舍。训练期空白一致性proxy未保证所有表示变体。
4. 完全无人工介入未实现：删除了手工命令词表/AST mask/逐式annotation，但reward weights、候选生成、统计特征、budget、dev utility仍有设计者选择。应说减少人工格式规则而非human-free。
5. 新算法是离线contextual merge policy gradient，不是原论文LM内部gater复现、不是已实现joint长期vocabulary/path RL。full-expected estimator让不同seed分词相同，属于重复确定性结果，不能算三次独立统计发现。

## 极端样本与稳定性

所有已运行54个method/scenario/suite/seed均100%无损，106797条原始结果。最长机械式2048项/71507字节，RL31820 token vsHF31582；深256层式RL/HF都是769 token；8193字节巨大数字RL/HF均8192 token，说明没有未知长数字单token魔法。
未知macro+Unicode：RL与HF都是39 token且无损；malformed rac{x}{两者都是3 token且无损。仅表示decoder与fallback健壮，不代表坏TeX有数学语义。
legacy v3/v4新paper参照：v3较短且命令更完整但仅约1.4MiB/s，v4严格zero-cross但较长约1.3MiB/s。因为旧模型训练数据不同、不构成同train公平排名；它们也见过target domains，不能算extreme域排除模型。

## 下一步研究应怎么定位

这个原型证明“RL离线生成标准merge table→成熟Rust在线编码”能解决速度等级问题，新奖励在未见paper上改善压缩/命令proxy。但当前主要未解是RL相对reward-greedy价值，及不用手工hard结构如何维持树完整性。
若继续此路线，应将RL目标推进为长期合并/vocabulary budget规划，使用多步actual rollout和泛化稳定reward，同时设立低人工语法合法性模块作为可选支路；不能在当前新test上反复调到更好后仍称未见test。
没有训练下游模型，所有泛化结论限于tokenizer。数字/结构/entropy指标不是数学推理能力。
