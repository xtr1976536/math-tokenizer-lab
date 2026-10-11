# 第三轮：固定语料的AST/RL优化

数学语料不变：corpus SHA256 `4ffe6202753593a295c981d665c60a47294a11d012acb579cb48dd68c3768733`；所有train/dev/test ID同v2。没有重新抓数据、修改标签或重训成熟基准。现有自研v2方法重测仅用于同输入速度对照。

## 为什么先改变实现与词表

v2 RL在每个原字节位置构造numpy数组、计算许多小softmax，反复进行树区间比较。原生C trie/DP复用已有策略参数与词表，清语法缓存后的全部dev推断逐token一致，约9–12倍提速；这部分只是一致性优化，不能当新RL质量贡献。

AST-v2从字节学命令，词表预算经常花在`\\fr`等命令片段而非完整运算符。v3从相同训练语料选择常见LaTeX命令原子，256个字节回退和原子均计入词表预算；然后受semantic AST/CST源区间约束的BPE学剩余项。保存训练命令列表，不能从test挑命令。

推断用固定词表的结构合法最短路径；因此该分支更准确叫atomic-command AST-BPE+structured DP。它改变BPE ranked merge推断，词表/推断/原生加速分开解释。完整decode维持原UTF8，未知符号始终保留byte覆盖。

## RL-v3

先用atomic-command BPE词表代替字节初始词表，然后训练完整路径REINFORCE。仍保留原REINFORCE、RLOO/relative基线，语言模型完全未接入。C原生实现执行学习到的相同局部归一化policy-MAP；soft结构罚项作为独立推断目标参数，只按dev选择。

候选：command原子64/128/192，各seed17/29；RLOO6epoch、RLOO24epoch、relative12epoch、RLOO数学符号reward；交叉罚项0/.5/1.5。压缩配置约束命令/数字保留>=v2 dev、树交叉<=v2 dev。全配置日志保留，不能只报告最好结果。选定后seed17/29/43共同test。

所有策略、原始reward/epoch日志、词表、source C SHA、native模式都保存。新词表no-RL shortest路径控制区分“RL训练”与“更好的词表/推断”。这种压缩目标已有DP下界，不宣称RL胜过相同可用token边的真正最短路径。

## 额外尝试

BPE大词表种子→minimum-path usage分阶段修剪，参考[PathPiece](https://aclanthology.org/2024.emnlp-main.40/)和[MinGram](https://arxiv.org/abs/2606.27019)。数学版本加入命令原子与结构约束，是本项目的适配，不冒充官方复现。探索结果未明显优于原子BPE，因此保留为探索，不强行选入主结果。

RL-vocabulary swap也是单列研究：train按文档内部留出，采样候选词表交换，用RLOO学习proposal策略并保留实际reward改善的incumbent。这个组合包含“随机策略+保守搜索”，不能把搜索收益全部称RL贡献。探索初版在dev提升很小，尚未作为正式主方法。源码与实验记录公开保留。

## 公平测量和边界

词表预算512/1024，原test已有开发暴露，配置选择严格仅dev，不称新未见holdout。原生kernel加速所有新方法；Python parser仍实际计入cold encode时间。速度报告每轮清structure cache，5次中位数；首次编译不包含稳态编码，但记录编译依赖。不能比较一个方法缓存而另一方法重解析。

比较压缩、command/number/lexeme、tree crossing、Rényi/entropy、vocab使用、稀有token、p95长度、编码速度与训练秒。多指标有冲突：更完整命令可能不提高词表利用率或Rényi效率。报告所有方向，不给所有指标同时更好的虚假保证。重点改进需同时量化长度、结构和速度。

native source由本项目编写，平台需C编译器；二进制不上传，源码hash自动本地编译，ctypes没有Python扩展ABI依赖。保持v2代码原样便于差分测试。

语法前端加速版本structure_fast在全部28347条语料逐条与v2的kind/spans/lexemes/depth/reason完全一致。native树兼容表进一步由逐候选遍历所有span改为端点crossmask预计算，任意非层状span随机测试也与参考compatible一致。
