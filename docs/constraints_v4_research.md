# 第四轮：约束与RL主线的文献及设计

## 文献证据

1. Huang & Ontañón, [A Closer Look at Invalid Action Masking in Policy Gradient Algorithms](https://arxiv.org/abs/2006.14171), FLAIRS2022：在策略梯度中给非法动作加mask，限定策略归一化的合法集合。本文支持invalid-action masking的一般动机，数学tokenizer适配与其收益需本地验证。
2. Ciosek & Whiteson, [Expected Policy Gradients](https://arxiv.org/abs/1706.05374), AAAI2018：对动作积分/求期望以减小策略梯度估计噪声。本项目有限无环分词图可额外对整个trajectory精确求期望；这是本项目实现，不是论文代码复现。
3. Dauncey & Wattenhofer, [You Can Learn Tokenization End-to-End with Reinforcement Learning](https://arxiv.org/abs/2602.13940), ICML2026：离散边界、score-function、相对reward与time discount减方差。本项目维持RL策略训练，而非把RL名字贴在shortest-path上；没有下游LM，所以reward是独立tokenizer压缩成本，绝不伪造early-exit CE。
4. Chen et al., [Language Model is Suitable for Correction of Handwritten Mathematical Expressions Recognition](https://aclanthology.org/2023.emnlp-main.247/), EMNLP2023：显式LaTeX结构符号、字母等最小符号的性质支持语法原子保留，但它研究HMER，不直接证明本tokenizer的收益。

## 什么可以作为硬约束

- 不部分跨切已识别semantic AST/CST节点：合法区间必须不相交、包含或被包含，训练与推断使用相同mask。
- 不在词表覆盖的控制序列内部切边界：`\\alpha`、`\\frac`等完整命令，包括转义控制符。命令本身必须是词表条目才能做单ID原子；BPE也可合并整个原子及其他完整节点。
- 不在词表覆盖的UTF8 codepoint内部切边界：多字节数学字母按完整字符处理；完整字符缺词表项时明确byte escape。
- 数字原子性可作为可选实验：任意长实数/科学计数不可能全部有固定ID；因此不能对无限未知数字作“一token数字”的保证。当前主约束以commands+UTF8为主，数字可选commands_numbers单列测试。

不直接把所有完整公式当单ID：词表大小固定，未知科研公式无限。整个式子是解析与样本边界，不能跨越两条公式，但内部可由合法原子/子树token构成。这样才保留泛化和词表预算可比性。

树未知、宏未知、codepoint未覆盖时：记录escape原因，保留原字节无损，不删除test样本；树crossing指标仅对同一parser成功的样本有意义，不冒称任意未知TeX的语义正确性。

## RL为何仍然是主线

第四版同时测试零长度先验初始化和1.5长度先验初始化的策略，在有限合法分词DAG上做on-policy期望成本梯度。设Vπ(s)=Σ_a πθ(a|s)[c(s,a)+Vπ(s')]，ρπ(s)为从起点到s的状态概率，则

∇θ J = Σ_s ρπ(s) Σ_a πθ(a|s)(Qπ(s,a)-Vπ(s))∇θ logπθ(a|s)。

代码执行policy evaluation和policy gradient ascent on negative cost，既没有使用最短路径标签，也没有用DP最优解监督策略。DP求的是当前随机策略的期望，非min Q；这是对REINFORCE的全期望/Rao-Blackwellized估计。entropy项为局部占用加权探索正则；本项目核心梯度用entropy=0的有限差分严格核验，含entropy项不冒称完整轨迹熵的全梯度。

保留：旧v3原策略、v3仅mask控制、零策略无更新控制、same-vocab shortest-path控制。零策略可能在确定性MAP/greedy中因同分选择变成很多byte token，因此同时报告uniform policy的expected return训练曲线，不把这个较弱no-update控制当成熟基准。RL与shortest性能接近意味着学到了合法分词，不证明RL能超过同词表最短路径的压缩下界。

## 人工干预的边界

硬约束来自可解释的格式/语法合法性，未手工指定每条公式如何分词。全部命令/codepoint原子从训练语料自动提取按频率分配预算；没有人工按学科排重要性。验证集选择原子预算与少量epoch/lr配置，仍是人的设计选择；不能称完全无人干预。目的在于让RL自动学习合法边内的决策，逐步减少公式级规则。

训练与推断一套mask、保留zero-init训练曲线和参数、source C hash、词表预算和逃逸记录。所有语料与划分保持不变、成熟基准不重训。当前test已开发暴露；结果不当作最终未见出版test。

原子预算的选择规则本轮偏向合法性：dev命令完整率达到99.9%的候选中再比较压缩。此阈值是公开的人为实验目标，不叫RL自行发现阈值；模型内的动作决策由训练学习。更小原子预算、更短序列的Pareto候选全部保留在dev日志，不能假装99.9%方案同时压缩最优。

有限分词DAG在语法约束较强时有大量单合法动作，梯度为零是预期现象；只有多个合法token组合的状态提供学习信号。不同超参数得到相同分词不代表没有训练，需读参数/expected-return曲线与zero-policy控制；也不能把这些确定性相同结果视为多次独立统计发现。

速度实测：硬约束预处理增加少量边界mask成本，第四轮未比第三轮进一步提速；保留第三轮原生加速优势，但本轮优先优化结构安全性。1024匹配先验控制揭示policy-MAP推断和expected-cost训练可能错位；不能把RL期望收益改善当部署分词必然变短。后续可研究entropy退火、价值目标学习、策略导出的确定性解码，但仍需保持真实策略训练/对照，不能直接换成最短路径冒充RL。
