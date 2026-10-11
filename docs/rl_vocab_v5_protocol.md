# 新一轮：RL学习词表、复合奖励、未见论文与极端泛化

## 决策与RL身份

从固定词表路径学习转回离线词表/合并学习。策略在训练语料产生的相邻merge候选中选择动作，并跨合并步骤共享policy参数；训练完成导出标准byte BPE merge table，推断直接调用未修改的HuggingFace Rust实现，不运行RL网络/AST parser。工程上以成熟实现对齐速度，不通过缓存预分词结果制造加速。

正式RL是contextual merge policy-gradient，当前仍对立即多目标候选收益学习，不冒称完整长期RL规划。候选来源与观测统计均来自train，包含频率、heldout code-cost、边界碎片、空白变体一致性。默认用有限候选的全期望score-function梯度，sampling estimator作为可选代码分支，不伪造teacher target。

## 奖励不能只是短序列

奖励含五项：heldout token减少；fit估计unigram code cost的heldout改善及词表payload开销proxy；泛用词法单元内部边界减少；fit稀有token的heldout暴露罚项；原串/无空白变体gain一致性。reward weights公开。它是tokenizer-only proxy，unigram coding cost不是语言模型BPB，边界保存不是数学理解，空白移除不对所有TeX保证语义不变。

fit/reward按paper group在train内部再切分，reward最多384式、fit最多6000式，记录全部ID。候选最多12，每step24policy更新。Length-only是消融，不选为正式主配置。frequency/random/reward-greedy控制相同内部fit语料，能分开RL更新和多指标贪心的价值。

Dev选择规则固定：document-macro tokens/byte+.15*(1-command integrity)，两种子17/29平均；只在四个composite配置中选主方法。正式seed17/29/43。这个规则仍是设计者选择，不称无人工参与；不手工排命令重要性，不加人工命令词表，不逐式annotation。

## 语料与新benchmark

新arXiv按官方OAI discovery、固定版本abs页license核验、CC BY/CC0限制，单连接4s，源码只读不执行。保持v2原数据/hash和原暴露test单独冻结，新增v5 release。题材及公式类型双字段（paper_domain/domain），complexity沿用固定两档，来源、转换、版本、行号、许可证保留。

Fresh-paper-test由新增论文group哈希预先分配，不把同论文片段分给train/dev。对全部旧release及新train/dev做规范去空白exact去重和符号重命名/数值折叠的syntax template去重。它比exact更保守，仍不是数学语义等价去重。旧corpus存在质量欠缺的记录未静默修正，v5新增稿件额外拒绝\\数字等可疑控制码，reject记录单列；自动审查不是数学专家逐条真值验证。

两个训练场景：broad=train/dev；specialized=仅PDE/ODE/analysis，剔除任何target域paper和对应cross-list类别文档。Extreme-domain-test来自新增论文里的algebra/geometry-topology/number-theory/combinatorics/logic，subject category不出现在specialized fit/reward/dev。broad对这个集合的结果不作为“完全没见过该领域”主张，因为broad训练已含这些领域。

Synthetic stress仅从fresh heldout公式及机械极端样本产生：去空白、字母双射重命名、2倍组合、长到2048项、深度256、未知macro/rareUnicode/huge数字/损坏TeX。origin标synthetic，不混入natural human corpus，也不在reward/dev中调参。malformed测试是鲁棒性/无损测试，不作有效数学结构评价。

protocol写全部test哈希而不读test文本；完成两场景dev selection并写selection hash后才读任何新test。首轮结果保留，包括不改善、失败或极端退化，不按test调整。

## 核心指标与速度

使用共同原UTF8，无特殊token、截断、按模型删数据。token/byte、command/number/lexeme、树crossing、Rényi、vocab使用、unigram bits/byte、UTF8完整性、尾部分位和encode速度全部报告。泛化要读“未见论文”“从PDE迁到未见研究领域”“机械极端”三层，不把旧分层结果假称extreme OOD。

普通HF BPE、相同RL-fit子集HF BPE、SentencePiece BPE/Unigram与两预训练参考使用同测试输入；成熟模型不同词表单列。速度以相同公开per-formula encode、同批次5次中位；RLBPE和HF共用官方Rust backend。没有parser参与RLBPE推断；树指标在encode外统一计算，所以不能把更少指标开销当编码提速。

压缩/结构算法改变可能提高某些指标、降低另一些指标；不许把复合reward=全部指标优秀。没有下游模型，不能把intrinsic或OOD分词指标当推理泛化能力。

历史v3/v4模型不重训练、不按v5选择，单列prior_data_reference测fresh新论文与目标domain。因为它们在旧训练中见过目标领域，不能把其extreme集合成绩解释为完全未见领域泛化。新specialized模型才符合target-domain exclusion。

正式主测试仅1024词表（参数扫在旧dev/probe完成）；没有按fresh test挑512/1024更好的预算。两场景20条dev config/seed记录；expected梯度让相同配置不同seed结果相同是正常，报告为确定性重复不是额外独立证据。候选reward fully observed的bandit也有直接reward_greedy强控制，如果它更好必须原样报告，不能因为“RL主线”而隐藏这个控制。
