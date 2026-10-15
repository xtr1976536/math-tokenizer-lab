# 数学分词器：四类泛化实验（2026-10-15）

## 原始全量记录（历史）

本轮固定训练型词表为1024，共9个模型、11组设置、99组模型评价。全量测试882,738条模型—公式记录，涉及51,598条不同测试公式。语料目录保持原始哈希。

模型：标准硬AST-BPE的数学迁移、AST-v4、Boundary AST、Math-atom BPE；五个基准为Byte BPE、SP Unigram、Qwen2.5、Llama 3与o200k。三个通用LLM编码保留官方完整ID空间；其余均1024。

四类测试：全领域来源隔离的三个划分（17/29/43）；三个学科留出组合；分析/方程与代数/数论的双向极限OOD；三个简单训练到复杂测试的来源划分。论文或资源模块作为来源组。跨集合ID、来源组、模板、MathML渲染重复均已排除。训练只用于词表/合并规则/Unigram概率；验证保留并执行同批接口诊断，参数在测试前统一固定。

八项指标：来源组等权T/B、Cmd、Num、完整encode接口MiB/s、Used、Cross、复合AST双端对齐AST-A、Rényi效率(alpha=2.5)。AST-A借鉴TokEval的节点双端判据并将评价对象扩展为数学复合节点；Rényi使用公开熵函数核对，以声明完整ID空间归一化。

## 复现

```bash
.venv/bin/python scripts/run_intrinsic_2026.py
.venv/bin/python scripts/audit_intrinsic_2026.py
.venv/bin/python scripts/build_intrinsic_2026_report.py
```

已完成的设置自动跳过；`protocol.json`包含全部ID和源代码哈希。每个设置包含models、summary.json、逐式压缩存档与4轮审查，`independent_audit.json`为第5轮独立复算。7项相关单元验证通过（4项新增与3项Boundary测试）。

`metrics.csv/json`为全部99组结果；`paper/intrinsic_20261015`保存中文稿件和三张矢量图；最终PDF位于`output/pdf/数学专用分词器_四类泛化实验_中文报告_20261015.pdf`。旧的词表扫描与这轮实验分别存储。主结果：Boundary AST在11组均有最低T/B；硬AST路线的复合结构对齐更高；AST-v4编码速度领先。完整数字、分指标排序和来源组bootstrap区间见PDF。

## 中文论文修订版

当前论文使用ICML 2026官方preprint双栏模板，展示AST-BPE、软AST-BPE、数学BPE与五个基准，共8个模型；指标为T/B、Cmd、Num、Cross、Used和编码速度。四类实验及每次划分的原始测量值保持不变。原稿与生成脚本保存在`history/papers/intrinsic_20261015_v1`，原始99组实验记录保存在本目录。当前展示88组记录、784,656条模型—公式评价。

修订版：`output/pdf/数学专用分词器_四类泛化实验_中文报告_20261015_修订版.pdf`。生成脚本仍为`scripts/build_intrinsic_2026_report.py`。
