# 研究协议 v1（2026-10-10）

## 项目范围与完成边界

这是完整大项目的第一阶段：可追溯的小规模公式语料、独立分词器实现及本地 tokenizer-only 测评。没有训练或连接 GPT-2 等下游模型，没有完成 arXiv 批量采集，也没有证明下游功能提升。后续工作必须保留已有结果及版本，不能用挑选种子／词表预算的方式更新“正式结果”。

初版语料不追求量大：EqGPT 全部可映射的结构、12 个 OpenAI 数学稿件的源代码、SymPy 历史文档、明确由助手生成的小规模模板。每篇文档最多保留按 ID 哈希排序的 120 个公式。人工／AI 归属是独立研究因素，需要来源方证据，不能用 AI 检测器推断。

## 分类与来源

一级分为 PDE 与 non-PDE。实际文件以 pde、ode、algebra、linear_algebra、analysis、number_theory、combinatorics、probability、logic、mathematical_physics 分类；后续新增 geometry_topology、optimization、statistics 等类别时独立版本化，不能把没有分类的数学片段随意装入 PDE。

领域标签包括上下文分类与保守规则。PDE 稿件中的范数、函数等一般表达式归 analysis；只有在 PDE 稿件中含等号和显式偏导／空间算子者进入 PDE。EqGPT 的 coefficient_free_pde_structure 是独立子类型：它是微分表达式结构，缺失原始系数与正负号，绝不补写“=0”假装还原原始方程。该子类型的 PDE 名称与来源采用上游声明，尚未逐条由数学专家复核；统计存在疑义的单变量微分结构需在下一版确认。

origin 字段分四种：

- human：EqGPT 论文明确描述专家人工整理结构；这只担保数学结构来源。LaTeX 映射代码由助手编写并机械执行，latex_author_origin 为 ai_assisted_conversion，不能宣称每条 LaTeX 都由人敲入。
- possible_mixed：OpenAI 官方声明内部模型产出稿件，同时明确有人工编辑例外。缺乏逐公式无人工干预声明，统一保守处理。
- ai_generated：本助手编写模板及机械参数扩展。记录模板编号与参数；不是独立模型随机采样，不能把参数扩展的相关数据当作独立样本。
- unknown：SymPy 历史文档。维护者是人、历史版本较早不构成纯人工无 AI 的充分证据。后续寻求可证实的来源；未知桶不参与纯人工子群结论。

原始内容生成来源、LaTeX 书写／转换方式、领域与正确性是不同维度，不互相推断。数学公式语法筛选不验证定理或等式真伪。OpenAI 稿件的结论未经本项目证明，不作为正确性 oracle。

EqGPT 论文报告 221 个 PDE，但固定版本工作簿含多个工作表，数量不相同。实际保留 229 条去重结构；拒绝无法映射的复合项、异常结构，所有位置与原串可追溯。不能把这个数量描述成“论文原始 221 条的无差异复现”。

## 公式筛选

只提取 TeX 数学定界符／单公式环境、RST 的 math role/directive 和明确 GRE 结构。禁止执行 TeX、\input、脚本或附带文件内的任何指令。拒绝未知宏、自然语言 \text、标签、注释、宏定义、未平衡大括号、孤立符号、超长段落。矩阵／align 系统等需要手工或专门解析的内容留在 quarantine，不能自动拆成失去语义的短行。

curated 包含公式与元数据；formula_only 每行只有一个 JSON 字符串（LaTeX 公式），模型输入只读取 latex 字段；quarantine 包含被拒绝原内容与原因，绝不是训练语料。上游全文和参考论文不上传为训练语料。

自动质量检查并不等于逐条专家审查。下一版须重点人工复核领域标签、疑似孤立上标标签、宏含义、导数结构及数学歧义；未完成部分在状态文件公开记录。

## 去重与集合划分

规范哈希只移除空白，不重写公式、不交换项、不化简。EqGPT 结构组将顶层加项排序作为家族标签；相关稿件按去除日期后的稿件名分组。合成数据同一模板所有参数归同组。固定 SHA-256 文档／家族哈希切分，70%/15%/15% 是组分配概率，不保证公式数量比例。

同一规范公式有多个来源时保留全部来源证据，跨集合的重复公式在公式层隔离，避免 x=0 等常见公式把整个文档图连接为一个巨大组件。未完成跨文档近重复语义去重，故不声称数据完全不存在语义泄漏；当前结果条件于这一版划分。

## 三种独立方法

AST-BPE：受限 LaTeX 语义表达式 parser -> 保留源字节跨度的 AST -> byte BPE 合并。合并区间与任一树节点必须不相交、包含或被包含，禁止部分跨切完整子树。仍可能在命令叶子内部保留字节片段。支持基本二元／隐式乘法、幂／下标、分组、分式、平方根和部分函数；尚不支持通用 TeX、求和／积分绑定、复杂矩阵和用户宏。解析失败整式回退字节，不能从测试中删除失败样本。

RL-BPE：每轮在训练拟合集上取 top-24 邻接合并候选，从训练集内部 20% 留出公式计算非重叠替换产生的 token 减少。线性策略在候选特征上用 categorical REINFORCE 学习选择，动作无关的期望奖励为 baseline，48 次采样更新后确定性选最高策略 logit。候选集合／已合并语料随轮次更新，策略跨轮共享。学习率 0.08，gamma=1，不使用语言模型，不访问 dev/test。

RL-AST-BPE：相同策略梯度、候选预算、奖励及更新次数；候选合并受相同 AST 源跨度约束。

这些是初始研究实现，特别是 RL 属 contextual one-step merge policy，奖励针对立即压缩而非长期序列决策，更不优化语义或下游损失。必须添加长期收益／MDL／语法原子保护等奖励与 ablation 后再论证新方法贡献。

Dauncey & Wattenhofer (ICML 2026) 学习 autoregressive U-net 内部动态门控，奖励来自 next-byte CE / early-exit 差值，结合 gamma=0.99 与 batch-relative advantage。其动态 latent token 不等同静态 BPE ID。本项目借用 score-function 技术，不复制其训练目标、方差保证或语言模型结论。官方仓库原样克隆并固定 commit；本机未执行其 CUDA/Triton/H-Net 训练，未列作已跑通 tokenizer-only 基准。

## 公平测评

训练组总词表预算相同，完整 byte fallback 必须计入。记录实际词表、特殊 token、模型工件哈希、输入 ID、训练用时。HF ByteLevel BPE 不用 regex 预切分；SentencePiece BPE/Unigram 用 identity normalization、byte fallback、无 dummy prefix、保留空白、单线程，未改第三方源码。

所有方法使用共同的原始测试字符串，无特殊 token、无截断、无数据依方法过滤；都检查精确 UTF-8 还原。评估函数相同。编码速度包含 AST 解析，只测 encode，decode 和指标计算除外；同一进程 5 次取中位数。独立性能脚本分别计时优化前后，实现差异与 Rust/Python 的差异不可混同。

纯字节、经典 byte BPE、HF BPE、SentencePiece BPE/Unigram 是算法基准。GPT-2、cl100k_base、o200k_base、Qwen2.5 官方 tokenizer 是成熟预训练参照，原始实现及资产不改；词表规模和训练语料不同，不能公平地据此声称算法优劣。固定词表的 raw-string 指标仍相同，但训练条件不匹配，报告里明确标组。

RL 各自使用内部 fit/reward 切分；普通 BPE 原生使用整个 train，另外给 RL 添加完全相同 80% fit 子集的 no-RL 控制。报告完整数据基准与内部控制两个比较，避免把少用训练数据的差异误称 RL 效果。各算法训练算力不同，记录而不冒称等算力。

测试指标：tokens/formula、bytes/token、精确无损率、AST 解析覆盖率、可解析子集的跨子树 token 比例、端到端编码速度、训练秒数、实际 vocab。任何 byte baseline 天然不会跨树，但压缩最低；零跨树指标本身不证明语义更好。没有 LM，不计算 perplexity/BPB 或下游准确率。

差值使用配对文档／家族 bootstrap 1000 次，bootstrap seed=20261010；每个训练种子的条件置信区间分开报告。自研种子 17、29、43，确定性算法实际相同是预期现象；不把重复确定性运行算作三次独立随机试验。

正式 pilot 预算384；512 的敏感性实验为固定次级设置，不根据 test 选优模型，不把最高一次得分作为正式结果。所有结果保留，不能覆盖完成的 run。

## 后续优先顺序

1. 补足可证实的非 PDE 人工 LaTeX，逐条来源证明；人工和 AI 的配对内容样本，同一主题／长度分层，避免来源与领域混杂。
2. AST 扩展到数学绑定、导数、矩阵、集合、宏表，建立黄金 AST 样本与数学专家复核，单列 parsing correctness 而不是只看覆盖率。
3. RL 长期候选序列与泛化目标实验，等 vocab／等 bytes／等 compute 的多维控制，独立无 RL、随机策略、reward-only greedy 等消融。
4. tokenizer-only 新论文基准兼容性调查；论文内模型的 tokenizer 不强行改成 standalone。算法源码尽量保持不变。
5. 最后进行 arXiv source 采集：使用官方下载接口和许可记录、限速缓存、固定版本、小批量；只源代码提取，不 OCR 猜 LaTeX。按论文与公式家族防泄漏。
6. 后续下游任务另立协议并取得用户对具体任务规模的指导，现阶段独立分词器不接模型。
