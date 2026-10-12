# 技术路线回溯：实际实现、目标与证据

审计日期：2026-10-11。审计基准commit：c640845。这里只回溯，没有修改模型、训练目标、原始实验结果或语料。结果证据来自源码、保存checkpoint、protocol/selection、逐公式结果独立重算，不能仅依据此前报告措辞。

## 核心判断

项目目前不是一个算法从v1到v4不断“全面升级”，而是几条路线发生分叉和合流：结构约束的词表/分词路线、固定词表上的RL路径学习路线、RL词表/合并学习探索、等价工程加速路线。v3保留压缩优先方案，v4保留结构合法性优先方案；v4已经与RL-AST路线合流。名字仍叫RL-BPE不表示它没有AST/CST。

最确定的收益来自结构回退覆盖、命令原子词表和等价原生推断。RL是真实policy optimization，但独立增益尚未充分建立。不能说hard masking的零crossing是RL自己学出的数学规则；不能说同词表shortest控制的压缩能力是RL贡献。

## 版本变化：表示、词表、策略与实现应分开

| 阶段 | 表示与合法性 | 词表如何学 | 选择分词/动作 | 核心实现 | 路线含义 |
|---|---|---|---|---|---|
| v1 AST | 受限semantic AST源跨度；解析失败整式byte | byte BPE频率合并，禁止partial crossing | ranked BPE | Python扫描/堆 | AST约束的原LaTeX BPE，不是AST序列BPE |
| v1 RL | byte，或同AST约束 | 策略在top24 merge候选选下一合并；train内部20%reward | contextual one-step REINFORCE，随后ranked BPE | Python | RL学习合并表/词表，后来未作为主线继续 |
| v2 AST | semantic AST +明确CST回退 | 相同源跨度约束的byte BPE | ranked merge | 增量索引/堆 | 表示覆盖升级，非通用semantic AST |
| v2 RL | byte，或AST/CST约束 | BPE先学固定词表，RL不改词表 | 完整分词路径采样REINFORCE/RLOO/relative；推断policy-MAP | Python trie/NumPy/DP | 学习对象从合并表切换为分词路径 |
| v3 AST | AST/CST hard spans +命令原子起点 | 自动抽train常见命令，原子+BPE | 结构合法shortest path | 原生C +等价parser | 词表改良+解码变化+工程加速组合 |
| v3 RL | 原子词表；byte分支soft crossing，结构分支hard | atomic BPE词表仍在RL之外 | 采样policy gradient训练，MAP加dev选择cross penalty | 原生C | 主要收益与词表/推断共同作用 |
| v4 AST | hard tree +covered command/codepoint边界 | 训练语料grammar atoms+结构BPE | constrained shortest path | C action masks | 合法性优先的非RL参照 |
| v4 RL | 与v4 AST同hard masks | 同样固定结构BPE词表，RL不改词表 | 期望on-policy gradient；greedy或MAP | 原生C mask解码 | RL与AST合流，真实RL策略+外部语法合法性 |

“树约束”=token源区间与每节点不相交、包含或被包含。允许token位于一个叶子内部，故v1/v2树合法不等于命令完整。“命令原子”=不能在词表覆盖的控制序列内部切边界，允许包含完整原子的大token。矩阵/积分绑定等的数学语义未统一完整解决，CST只保留显式LaTeX结构。

## 路线一：结构感知与命令原子词表

目标：保留数学单位及嵌套关系，在固定ID预算内短序列、完全无损。主线是“原公式→语义或语法源跨度→完整原子→结构合法词表与解码”，不是“AST序列化→AST节点BPE”的原始提案的完整实现。

当前未做：AST规范化、等价公式归一、用结构化节点ID代替原TeX、真正任意子树词典压缩。decode回原串是优势，但不同TeX表示仍是不同字符串；语义变化不由编码保证。

优势：hard tree零partial crossing可验证；命令原子大幅降低碎片；未知符号仍byte fallback；同词表预算完整计入256byte+原子。v3 AST比v2长式更短、词表更有效；v4命令完整率99.52%。

弱点：依赖人为parser与允许宏表，零crossing只相对本parser；命令/CST不等于数学语义。v4覆盖更多稀有命令占用预算，512词表有206多字节原子+256byte，只剩50个合并条目。更强结构合法性压缩代价真实存在。少量未覆盖宏/codepoint仍例外，不能保证所有未知单位单ID。

最重要的控制：v3/v4非RL constrained-shortest解码。它本身已经能提供良好结构/压缩，后续“加入RL”的优势必须超出它或优化它不能解决的目标。

源码：src/math_tokenizer/ast.py、structure_v2.py、structure_v3.py、ast_v3.py、atoms_v4.py、constrained_v4.py。

## 路线二：固定词表的RL路径学习

目标：固定词表中自主选择分词边界/下一token，减少人逐式指定分词；主线是状态→合法候选token→学习到的categorical policy→完整trajectory。

v2/v3学习器是7个共享线性参数的scorer，不是上下文Transformer/GNN，也不学习parser或vocab。特征包含token长度、train频率、词法边界、原子拆分、tree crossing、相对长度、一步最长token前瞻。正式policy更新只见hash选取的train子集（v2通常512、v3/v4最多768），词表训练见完整train。

v2/v3用REINFORCE采样、RLOO或relative baseline、entropy和discount消融，终止目标基本是负token成本。relative基线是固定byte成本，不是上传论文的early-exit LM reward；γ=.99是实验性信用分配，不把它称原论文理论完整复现。

v4利用有限DAG对当前policy的整条轨迹期望求值，再由forward occupancy计算cost policy gradient。不是最短路径监督、不是把argmin直接当policy，也不是完整LM端到端。RL cost-only主梯度经过finite difference，entropy项只是占用加权局部正则，未声称是完整trajectory entropy目标的全梯度。

优势：真实参数/expected return有改善、无需人工gold segmentation；低方差全期望适合可枚举有限状态。v2字节分支对匹配no-update初始化有小幅压缩改善。v3把policy编译后速度瓶颈明显改善。v4合法动作保证不会采样已定义非法边。

弱点：目标仍主要是token数，既不优化真实数学语义也不优化下游损失。固定词表+可枚举图+unit token cost已有shortest-path最优控制，RL没有理由在纯压缩下超过它。7参数容量很小，少量人工特征/规则仍主导设计。expected stochastic reward优化与greedy/MAP部署有错位；1024 matched initial控制略优于训练后确定性分词。v4 hard masks使crossing特征恒0，模型不需要学习这个规则。

“减少人工介入”当前实现的是无人工逐式分词标注，不是无人设计的tokenizer。parser、原子预算、特征、reward、学习率、dev选择阈值仍由人/本助手设定。

源码：rl_v2.py、rl_v4.py；v4对具体部署模式读模型顶层inference字段。1024训练config.infer默认greedy，实际顶层inference=policy_map；不能只读训练config误认部署算法。

## 路线三：RL-AST融合

目标：结构合法与自动分词同时成立。v1是AST合法合并候选+RL选merge；v2/v3是hard AST/CST合法路径+RL选token；v4原RL-BPE正式词表/动作都带hard tree，已与此融合主线重合。

优势：解析子集零crossing、无损、策略可自动学习合法动作内偏好，适合后续多目标学习。

弱点：当前结构图许多状态只有一个合法选择，RL容量与收益空间有限。v2 RL-AST不优于same-vocab no-update，v3 RL-AST与AST shortest接近但略长，v4又与shortest压缩几乎重合。不能把这类现象写成RL已发现更高层数学关系。

当前研究问题：RL在合法集合内还能学习什么，目标为什么超出unit-token shortest路径？必须具体定义，否则融合只是增加训练步骤但复制现有控制解。

## 路线四：RL合并表与词表优化——探索，不是当前正式模型

v1 RL合并选择：top24候选、4参数policy、立即holdout替换gain，跨轮policy共享。不同于完整长期规划；v2以后词表由BPE固定，这条原始研究线实质上暂停。

v3 minimum-path usage修剪：大BPE seed词表、按path使用gain分阶段剪到预算；借鉴一般tokenizer词表优化，是确定性启发式，不是RL。full-train 512 dev=18.724/18.811、1024=15.153/15.569，未超越原子BPE，未选主模型。

v3 RL词表交换：RLOO训练proposal scorer，采样删除/加入token，然后接受sample里实际reward改善的best incumbent；它是“RL proposal+保守搜索”的组合，非纯RL规划。byte探索dev18.361→18.295，AST18.931→18.984，变化小/泛化不稳定，仅dev探索。

目标：让RL学词表内容，使部署可用快速确定性编码，消除fixed-vocab shortest下界造成的RL冗余。潜力：更贴近用户“自动设计tokenizer”，部署与学习解耦。困难：动作空间和长期回报、候选发现、词表预算与byte fallback、同数据/搜索控制；尚未做joint vocabulary+segmentation有效正式验证。不能把它列成已成功新方法。

源码：bpe.py（v1）、vocab_v3.py（探索）。

## 路线五：等价工程加速——所有算法共享

目标：输出完全不变时减少时间和内存。演进：全扫描→受影响pair索引更新；ranked heap linked arrays；trie与C DP/softmax；树crossmask端点预计算；一次lexing/offset parser；v4合法动作mask和单候选跳过softmax。

证据：等价native v2→C在固定dev及test逐token核对；structure_fast在全28347公式kind/spans/lexemes/depth/reason一致；v3 RL约12倍旧Python推断速度。v4新增mask让速度相对v3略降，不能沿用v3比例宣称v4再次提速。

优势：收益可靠、checkpoint可直接使用。弱点：主要是Python→C及避免重复工作，不能作为RL算法独立质量贡献；仍依赖Pythonparser，未与成熟Rust基准同轮重跑，不能据旧速度表断言已超过它们。模型首次编译/加载不在稳态encode中，训练秒与编码秒分别列。

源码：legacy_fast.py、merge_v2.py、native_v3.py、native/decoder.c、native/decoder_v4.c、structure_v3.py。

## 同一语料上的主要结果（token/公式越小越短）

v1初始pilot语料1713条、test406条，与v2以后的28347/test3959不是同一实验；不能直接比较其指标为模型成长。下表v1方法用v2发布语料重新训练所得值，才可放入同表。v2-v4完整test IDs/hash一致。

| 方法机制 | 512 | 1024 | 应如何解释 |
|---|---:|---:|---|
| byte BPE控制 | 18.8636 | 15.0573 | 相同预算的算法参照 |
| v1受限AST+byte merge | 32.6042 | 31.8783 | 解析失败byte回退主导劣势 |
| v1 RL选merge | 19.2354 | 15.2983 | 未优于普通BPE |
| v2 hybrid AST/CST+ranked BPE | 20.1018 | 15.8032 | coverage更好，仍结构压缩代价 |
| v2 RL路径 | 18.6763 | 14.9190 | 对ordinary BPE小幅好 |
| v2同初始化no-update路径 | 18.7651 | 14.9836 | RL额外压缩约0.47%/0.43% |
| v2 same-vocab shortest | 18.6052 | 14.7239 | 仍比RL短 |
| v3原子AST+shortest | 18.5090 | 14.9752 | 原子词表+解码变化+native组合 |
| v3原子RL+soft tree | 17.2916 | 13.9361 | 当前RL压缩优先分支 |
| v3 same-new-vocab shortest | 17.0482 | 13.9172 | 比RL短；RL仍改善crossing等部分指标 |
| v3 RL-AST | 18.5289 | 15.0062 | 未胜过同分支非RLshortest |
| v4硬tree+covered atoms shortest | 19.4264 | 14.9227 | 结构合法性控制，99.52%命令完整 |
| v4 masked RL | 19.4269 | 14.9462 | zero-cross严格版；不是v3全面替代 |
| v4 matched initialization no update | 22.9520 | 14.9437 | 512显著学习，1024部署压缩无增益 |

v4 512相对v3 RL序列增加12.35%，1024增加7.25%，crossing降0；命令完整率分别86.83%→99.52%、96.50%→99.52%。v4速度约1.10/1.07MiB/s，v3同轮重测约1.16/1.15。v4 AST1024比v3微短0.0525 token/公式，但95%文档bootstrap[-0.2782,0.1361]跨0，不能宣称显著压缩提升。

## 新补的严格RL学习核验（只诊断，不训练）

之前的训练日志第一epoch平均混合了该epoch内多次更新，不能当作“真正初始policy”的objective。此次用保存的词表/频率，固定原768个train样本，重算matched-initial与最终policy的期望成本：

| 预算 | 初始期望token/公式 | 训练后 | 文档长度正规化cost初始→最终 | dev确定性分词发生变化 |
|---|---:|---:|---|---:|
| 512 | 20.7751 | 19.0664 | 0.61637→0.57814 | 2183/2808 |
| 1024 | 15.5980 | 13.8947 | 0.48967→0.44537 | 19/2808 |

结论：RL确实改变参数和随机policy、降低训练objective，但1024部署分词基本没改，仅19个dev式变化，test比matched initial多0.0025 token/公式。不是“没有RL”，而是“真实RL objective改善未转化部署优势”。zero-init 512 greedy tie会选较多byte，matched-initial弱控制不能替代shortest强控制。

模型实际只有7个共享线性参数。v4hard crossing feature恒0；512词法split feature在训练候选分支也恒0，合法动作mask减少了可学习自由度。硬树规则有作用，但作用与RL学习必须分开。

## 本次严格回溯发现的记录与归因问题

1. **v4实验数量此前说错。** 实际dev_ablations每budget63行=3 AST+60 policy/seed/decode记录；后者来自30次策略训练，各评估greedy/MAP，两个budget共60次policy训练，不是123行/预算。正式每budget14个test方法/seed、55426记录正确。原报告历史表述保留，本审计以artifact实际计数为准。
2. **v3每budget80条dev行不是80次训练。** 包括5 AST、3 noRL控制、24次policy训练×3 crossing penalty。两budget48次policy训练；不同罚项是同policy推断消融。
3. **v4实际原子预算256只保留206个多字节原子。** 另一个统计到的原子是单字节控制组合过滤结果；以checkpoint reserved_types=206为准。512因此剩50个BPE词表slots。不能称保留256个完整命令。
4. **速度与质量归因不能混用。** native exact输出相同证明速度优势，不证明RL更会分词。v3改词表、解码、特征、policy训练子集多项，不是只改变一个因素的因果消融。
5. **v4 RL与AST正式词表同源、合法图相同。** 主分支已经结构融合，无结构RL-BPE与RL-AST不再是三个彼此独立、均经过最新优化的模型。
6. **零cross指标定义依赖所用parser，并且由mask直接保证。** 它是满足硬约束的证据，不是数学语义正确性、推理能力或RL自动发现语法证据。
7. **v4单独“更新收益”控制只在17seed构建同频率initial一次，再用于29/43比较。** 当前freq选择确定性且样本一致，合理，但不能称三个独立强baseline训练；bootstrap对既定模型和既定test条件化。
8. **训练成本范围不统一。** v2 train_seconds有时只policy，v3另有vocab/policy/total，v4正式RL文件仍只policy；不能直接用这些字段排名总体训练效率。v4大epoch日志中的first epoch不是初始model objective，本次已独立诊断校正解释。
9. **v4精确梯度声明应限于cost主项。** entropy实现是局部occupancy加权正则，没有轨迹熵未来项的完整gradient；cost-onlyfinite-difference成立不证明全部regularized目标完全无偏。稀有token成本在正式所选配置为0，不能称已优化语义/rare使用目标。
10. **探索词表学习没有进入正式checkpoint。** 初版top24merge策略和v3swap存在源码，但v2-v4正式RL均没有用RL改变词表。joint optimization是未完成方向。

## 实验可信范围

独立重算8个正式目录、456602条已存逐公式记录：每模型输入ID一致、总token/byte/roundtrip与metrics一致；v2/v3/v4 test ID哈希和corpus hash一致。它证明结果存储及聚合一致，不证明研究设计所有方面正确。

v1与v2完整训练在全新checkout可重跑；v3/v4主要完成checkpoint重新编译/全test重放，未独立重训练全部候选。125项tests与若干gradient有限差分支持实现，非外部同行审查。

test从v2开始多轮开发可见且语料质量修正前部分test已看，protocol dev-only选择是流程规范，但不能消除跨轮adaptive test exposure。没有新的从未查看paper holdout，因此CI只能解释当前数据，不作正式未见泛化结论。

test来自70个document/family，bootstrap按组1000次是条件区间；种子常给相同deterministic分词，不当多次独立发现。复杂度/来源/领域分层有混杂/小组；近重复语义去重和expert gold AST仍缺。树前端和树指标同一来源，不能用自身合法性指标证明模型具备数学理解。

下游模型从未训练。fertility、Rényi、经验unigram bits/byte不是模型BPB，也不是最终提升各种下游功能的证明。固定大词表成熟tokenizer与小预算方法不在公平训练排名里。

## 下一次设计前应先作出的路线选择

- 工程/结构路线：保留v3压缩优先与v4严格约束两条可用分支，不再用版本号暗示后一版全面更好。
- RL路径路线：明确目标要超出“固定词表unit-token最短路径”，并对齐随机期望训练与greedy/MAP部署；否则RL会追踪已有最短路径控制。
- RL词表路线：如果核心目标是减少人介入tokenizer设计，应重启真正的词表/合并决策学习，但尚未有正式成功证据，不能提前宣称主线已完成。
- AST路线：先明确想学语义树/子树表示，还是只做源跨度安全分词。当前是后者，不能用AST名字掩盖通用semantic parse尚未做。

此处是路线审计结论，不是新算法、未经授权的下一阶段修改或整项目完成声明。
