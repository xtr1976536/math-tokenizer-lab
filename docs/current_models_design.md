# 当前两条路线：AST-v4与v5

本轮以用户指定的两个版本为基础，不再推进v1—v3或RL-v4训练模型。当前API为`math_tokenizer.current_models.load_current`，配置与路径由`models/current/registry.json`统一管理。旧命名的parser、字节序列、原生trie等模块仍被当前实现共用；它们是底层依赖，不是活动模型版本。

## AST-v4：相同约束，减少解码开销，改善词表

编码仍保留源LaTeX、语义AST/CST源跨度、覆盖命令/codepoint硬边界、未知单位byte escape。新`mt_shortest`只做单位token成本的受约束最短路径：遍历真实trie前缀时累计跨树条件，省去`n×max_token_length`跨树表、策略softmax、最长token前瞻和不使用的特征。Python端省去无用safe/interior数组，预存完整词法单元与Unicode覆盖，减少逐字符编码与逐词切片。

固定checkpoint在两个预算各54,551条自然/stress公式上与旧AST-v4逐ID一致。因此这里是等价工程优化，没有将更少约束或缓存预分词结果当成加速。

词表试了两种方式。第一种在BPE前直接占用子树ID，dev好而重复benchmark退化，保留结果但不用作默认。第二种保留原1024项BPE池，按fit中重复源子树提出额外payload，再用train内部独立论文的收益筛回1024项。256字节和原有完整语法单位受到保护；新增片段仍受原hard mask约束。

筛选分数是reward文档正规化的token使用量乘以“删除该token后，其独立payload的局部替代token成本减一”。替代成本是代理，不是所有真实上下文的精确全局损失。候选数64/128/256/512由dev比较，当前两场景均选择512个候选，最终只保留43/37个新增子树。并不额外扩词表。

这使一部分完整语义/语法子树可以直接用词表项表示，但仍不是通用semantic AST节点序列BPE，也不做变量重命名等语义归一。

## v5：离线学词表，在线使用原版Rust BPE

用户提出的“两阶段”在这里是明确的：离线阶段决定合并顺序和词表；在线阶段只加载固定表进行确定性编码。策略、critic、训练奖励及树parser都不进入v5在线编码。

“离线”指部署前的训练时段；完整轨迹分支在训练中采用on-policy采样。

原local阶段仍保留，并优化其实现。增量索引同时维护原始相邻频率、真正不重叠替换收益、词法边界减少量；`(a,a)`单独处理重叠，版本堆保持旧候选排序。这样不必每个候选都扫描reward文本。原训练方法得到逐项完全相同的词表、merge和11参数策略。

`MergeSession`统一提供候选统计、真实合并、预算和导出，local策略、完整轨迹策略与控制组使用同一环境。重复调用local `train`会重新初始化预算，避免把旧词表累积到新训练。

### 完整轨迹的RL学习对象

原v5每次优化立即候选收益，新分支在一条完整合并轨迹内冻结策略，用最后导出的真实Rust编码结果评估完整词表，再更新策略。策略有22个共享线性参数：原11项候选统计，以及它们与词表预算进度的交互。它可以学习不同预算阶段的动作偏好；不是上下文Transformer或GNN。

终局utility包含文档宏平均token/byte、fit估计的held-paper unigram编码量、命令与数字碎片、可测跨树比例、有限安全布局变体差异、fit稀有token暴露。编码概率使用Dirichlet平滑并由fit估计；它是unigram预测编码代理，不是下游LM BPB。布局探针保留文本宏及命令与字母之间的空格，不宣称通用TeX语义归一。

奖励相对同一起点策略的终局utility正规化。对于完整merge轨迹τ：

\[
g_i=(R_i-\overline{R}_{-i})\sum_t\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t}).
\]

使用3条独立轨迹的RLOO、3个epoch、γ=1、探索混合和Adam。当前公式只在每条轨迹结束后更新，不是在每个merge内做24次立即收益更新。最终用学习后的策略确定性生成部署合并表。随机期望训练与确定性表生成仍可能错位，因此保留相同初始化no-update与相同rollout数量的搜索控制。

### 学习未来回报的critic试验

进一步尝试25维状态统计的线性ridge value head。每条轨迹的baseline只用其他轨迹的终局回报拟合，当前轨迹不为自己的baseline提供目标；避免同样本critic引入的score-function偏差。其更新为：

\[
g_i=\sum_t\bigl(R_i-\widehat V_{-i}(s_{i,t})\bigr)\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t}).
\]

value head读取预算进度、当前token成本和候选统计均值/标准差，不读取gold分词或test。两步完整随机图上枚举所有成对轨迹，检查其梯度期望与有限差分一致。这个检查支持实现中的条件独立与导数，但不保证有限样本方差一定降低。

实际dev的12个batch中，相同轨迹下critic/scalar-RLOO坐标方差比均值1.012、中位1.015，未稳定降低。dev总体也没有胜过RLOO；因此保留研究代码和结果，默认部署不使用critic。当前通用模型选RLOO，专用模型选原local v5。

## 如何理解RL主线与人工参与

新增轨迹策略是真实score-function RL，critic的目标来自RL终局回报；没有最短路径teacher、逐式人工分词标签或隐藏test奖励。数学格式parser只在AST编码及v5离线reward/共同评测中使用，v5在线没有hard AST。

奖励、特征、候选提议、预算、采样配置、parser与dev选择仍有设计者选择。减少的是逐式标注与在线手工数学约束，尚未实现完全无人介入。no-update搜索控制使用相同的RL warm-start策略，故它检验新增轨迹更新的贡献，不能被称为整个流程完全无RL。

上传的[Dauncey与Wattenhofer论文](https://arxiv.org/abs/2602.13940)提供离散score-function、相对奖励和减方差的参考。本项目没有连接其语言模型gater，没有早退LM预测头，也没有把本文γ=1称为原论文γ=.99方案的复现。[RLOO相关公开研究](https://aclanthology.org/2024.acl-long.662/)用于估计器定位；交叉拟合value head是本项目独立实验。

## 当前入口与复现

```python
from math_tokenizer.current_models import load_current

strict = load_current("ast_v4", scenario="broad", capacity=1024)
fast = load_current("v5", scenario="broad", capacity=1024)
text = r"\frac{\partial u}{\partial t}=\Delta u"
assert strict.decode(strict.encode(text)) == text
assert fast.decode(fast.encode(text)) == text
```

AST-v4保留512的等价加速checkpoint；当前新词表试验及v5均以1024为主。`specialized`使用PDE/ODE/analysis训练场景，按原v5规则排除目标领域及cross-list论文。

```sh
# 完整顺序入口。使用一个尚不存在的输出目录。
sh scripts/run_current.sh results/my_current_run001

# 仅重放当前已保存的固定配置结果。
.venv/bin/python scripts/replay_current.py results/current_reproduce17
```

入口只训练选定的AST-v4/v5配置，不重建语料，不恢复旧模型，不重跑SentencePiece或预训练tokenizer。旧实验结果及旧脚本作为历史证据保留；旧模型重放需要清理前Git revision。当前结果见`docs/current_models_results.md`。
