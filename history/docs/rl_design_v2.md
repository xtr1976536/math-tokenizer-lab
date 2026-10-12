# RL与结构分词改进 v2

## 诊断与改动

v1的策略每次只选一次合并，特征少、立即奖励和泛化样本少；结构解析失败整式byte回退，容易让奖励被解析覆盖主导。第二版分开处理这两个问题。

结构前端保留原受限semantic AST，增加安全的LaTeX syntax tree（完整分组、命令叶子、矩阵环境等）；任何语法树回退单列，绝不虚增语义AST覆盖。hard结构合并用层状区间约束，增量候选索引只更新受影响的公式，避免每个vocab项扫描全部语料。v1实现和数据快照保持可复核。

RL-v2从标准BPE获得固定字节词表，给定原式枚举无损token DAG，策略在每个源字节位置的合法候选token中选择下一段，形成完整分词轨迹。词表容量、内容、基准路径和策略学习分离，便于判断改善来自哪里。它是“BPE候选词表上的RL分词”，不再称RL合并表训练。

收益为完整路径的负token成本，支持额外符号边界罚项。采样多条轨迹，沿源字节位置对齐return，策略更新不对reward反传。gamma=1针对完整压缩目标；gamma=.99单列偏置/方差消融，折扣按源字节距离计算。优势和基线不借用不存在的语言模型头。

## 与上传论文的对应和差别

[Dauncey & Wattenhofer, ICML 2026](https://arxiv.org/abs/2602.13940) 的贡献点包括early-exit相对reward、discount return、batch-relative centering，以及边界概率目标下采样率。公式(15)-(18)、(23)-(27)已从用户PDF核对。

本项目沿用离散score-function和减方差思路，实验：naive REINFORCE；action-independent字节逐个分词参照基线；leave-one-out多轨迹基线；按字节γ=.99；entropy系数；零初始化长度先验；更多epoch；软结构罚项。初始长token先验及no-RL控制公开，不能把这个先验产生的压缩当RL收益。

原论文early-exit reward依赖next-byte预测模型，不能在用户要求的独立tokenizer阶段伪造。当前“relative”只有固定byte cost baseline，不叫early-exit复现。若以后要优化概率描述长度，需要另行设计完整有归一化意义的压缩目标，而不是把token频率当LM预测。

多轨迹RLOO基线参考[Buy 4 REINFORCE Samples, Get a Baseline for Free!, 2019](https://wouterkool.github.io/publication/buy-4-samples-free-baseline/)与[Back to Basics, ACL 2024](https://aclanthology.org/2024.acl-long.662/)。每条轨迹基线排除自身，其他同公式轨迹独立采样。v2按字节位置对齐成本到终点，具体目标是文档长度正规化的policy路径收益。它不同于token-level语言模型。

## 预定实验设计

每个预算两分支：byte和hybrid AST/CST。byte10种配置、AST9种（hard约束不重复soft结构配置），各训练seed17/29；共38个dev消融。拟合词表使用完整相同train；策略更新在train按hash固定最多512公式，用6、12或24epoch、K=4或8独立轨迹、batch16、Adam、梯度裁剪。训练样本限制及算力差异公开。

按dev中固定文档均衡样本的宏平均token/byte选配置，名称作确定性同分打破；selection.json写入并hash冻结后才访问test。选定方法再用17/29/43，共同原式测试。两个预算512/1024是预定设置，绝不用test选择预算。

控制：原BPE、改进AST-BPE、固定同词表shortest path、相同无RL初始策略、旧v1方法、官方HF与SentencePiece、预训练参照。shortest path是固定词表／合法边的最少token参照，RL不应宣传胜过它的压缩下界。若RL和no-RL相同、劣于BPE或没有显著差异，都原样报告。

可用的后续改进方向：词表同时优化、长期合并收益、teacher shortest-path辅助监督或policy distillation（要明示不是纯RL）、更强非线性策略和数学结构奖励、完整MDL。当前先通过严格实验判断是否有意义，不以结果必须更好为前提。

策略特征增加一步前瞻：下一字节位置存在的最长词表token长度。它只使用原字符串与固定词表，不使用下游模型；该独立encoder能够读完整公式，不能称Dauncey严格因果门控的复现。无RL先验、same-vocab最短路径与RL分开比较。
