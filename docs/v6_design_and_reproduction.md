# v6：语料、两个模型家族和统一评测

主语料位于 `data/releases/v6/curated/corpus.jsonl.gz`，加载时自动解压并核验哈希。`formula_only/<domain>/<origin>/<simple|complex>.jsonl` 每行仅一个 LaTeX 字符串；来源、作者、许可证、源码位置、原始公式、变换、复杂度分量和独立参考树保存在主记录。文件和模型都不包含下游语言模型权重。

语料中的 human 沿用用户批准的 2020 年前 arXiv 历史来源规则，不等于逐条作者认证。Fungrim/SymPy 保留 unknown；OpenAI 官方稿件保守归 possible_mixed。AI 模板和 EqGPT 的系数自由结构串放在 supplements，不参与主库排名。AI 主分区不以基础模板补数。

Fungrim 主题按版本化模块表标注，交叉主题另列；例如 gcd/totient 是数论，modular 系列同时保留分析及几何标签。官方条目的 Variables/Assumptions 元数据一并保留，不能把有条件恒等式称作对任意变量普遍成立的定理。研究来源、语法有效和数学真理是不同证据。

所有 arXiv 下载为单连接、缓存优先、至少间隔四秒；仅接受明确 CC BY/CC0 的固定版本，403/429 停止。源码仅解包读取，不执行 TeX。有限参数宏使用 TeX 的直接参数替换，不能额外加花括号；所有接受的自定义宏公式经过未改动 KaTeX 的原始宏/展开后 MathML 一致性检查。公式尾部 `\,` 等控制符不会被句末标点清洗截断。

四轮质量审查分别覆盖原始文件及许可、独立语法及宏保真、分类与重复/划分、分层源码重放及纯公式文件。数学条件依赖和难度不作专家逐条认证。arXiv subject 是来源主题代理，领域标签不能保证数学概念完全不相交。

极端泛化训练/开发池仅接受主题可核验的 arXiv PDE/ODE/分析论文，排除数论、组合、逻辑、代数、几何以及相关交叉分类；混合公式库整体排除。正常广域场景保留各类合格来源。全部旧版本测试集视为已暴露开发资料；新论文测试避免与全部旧公式和新 fit/dev 重复的语法模板。

AST-v4 使用支持算术优先级、分组、上下标、分式和根号的原生表达式 AST；其他有效 LaTeX 使用明确标注的命令/参数/分组语法树回退。训练依结构合法的源区间执行 BPE，再用训练内部 held-paper 使用量及局部替代分词代价剪枝子树词表。上线使用同一结构约束的最短路径，全部 256 字节回退及保留命令/codepoint 都计入词表预算。它不是通用数学语义证明器。

v5 保持训练阶段学习策略、部署固定词表的两阶段路线。接受三条实验分支：完整合并轨迹 terminal RLOO；独立 KaTeX 树提供即时跨节点变化的精确期望策略梯度；RL 学习合并屏障、交给官方 Rust BPE 拟合词表。局部策略是上下文 bandit，不宣称长期全局最优。屏障策略如果收敛为空、与相同原子化频率 BPE 重合，必须报告无额外 RL 优势。

词法原子只从 fit 公式的控制序列与 Unicode 获取，权重/预算仍是显式设计选择。未知单位使用字节回退，不能承诺有限词表对任意新命令都保持一个 ID。部署均使用原版 tokenizers Rust BPE；原生 codec 只执行命令/codepoint 到固定符号的转换。紧凑别名缩短内部编码，不改变 primitive ID 或最终 token ID。CPP/Python 桥和 ctypes 参考实现逐条验证一致。

训练内部 fit/reward 文档隔离，所有同预算基准使用相同 fit ID 列表；独立概率读数也在相同 fit 上估计。1,024 预算比较全部分支、三种随机种子及关闭更新/频率/贪心/等预算搜索控制。2,048 继承测试前选择的配置，评估最终两个家族及相同经典基准，避免在测试集上重复调参。期望梯度分支是确定性的，三个 seed 文件不能当三次独立随机实验。

第三方基准源码没有修改：HF byte BPE、SentencePiece BPE/Unigram、字节封装 WordPiece、作者 PathPiece，以及官方 Qwen2.5/Qwen3/DeepSeek-V3/Phi-4/Mistral-Nemo 和 tiktoken GPT-2/cl100k/o200k、Meta原版Llama 3 tokenizer.py。Llama 3词表从公开发行镜像取得，完整文件Git blob与Meta官方记录一致；无损rank转换后通过47,773条主库和521条压力输入与HF配置逐ID核对。原版代码和LICENSE由fetch_pretrained_v6.py按固定revision/哈希取得，代码不修改，HF/Meta两种接口读数都披露，正式榜只计一次同一切分。WordPiece 的 UNK 用已计入预算的字节 escape 保持原文；长输入使用与原版逐 ID 核对过的原生最长匹配，避免二次子串扫描，词表和算法不变；PathPiece 库空串会 panic，接口对空串返回零 token，完整数学基准没有空串。成熟模型保留官方 normalizer，任何原文损失都记录，不通过改动官方配置隐藏问题。

四轮评分前公平性检查覆盖同数据访问范围、全部 ID 预算和原始接口、指标正负对照及官方熵定义、全模型重载和公开源码版本。全量评分后追加覆盖/独立汇总、全模型逐式重放、官方指标和七轮交错计时复查。没有按模型删式或截断输入。固定大词表预训练参考不能用于同训练预算的因果算法排名。

快速重放现成实验：

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-v6.lock.txt
uv pip install --python .venv/bin/python -e .
cd runtime/katex && npm ci --ignore-scripts && cd ../..
# 依 references/upstream_code.json 获取固定版本的 TokEval 和 PathPiece。
# PathPiece 需要 Rust/maturin 编译，源码无需修改。
.venv/bin/python scripts/fetch_pretrained_v6.py
.venv/bin/python scripts/audit_preflight_v6.py
.venv/bin/python scripts/stress_benchmark_v6.py
.venv/bin/python scripts/evaluate_full_v6.py
.venv/bin/python scripts/retime_full_v6.py
.venv/bin/python scripts/compare_pretrained_v6.py
.venv/bin/python scripts/audit_full_v6.py
.venv/bin/python scripts/summarize_full_v6.py
.venv/bin/python scripts/report_comparison_v6.py
```

不要把已经用于调试或调参的测试集再称为全新未见。源码审查发现的宏错误及领域排除错误有明确作废记录；受影响权重已删除，源码规则修改独立于测试成绩，正式实验使用修正后的唯一版本及固定配置。
