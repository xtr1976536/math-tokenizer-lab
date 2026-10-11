# v5 benchmark质量与可用范围

来源、版权、公式质量和数学正确性分开记录。新增稿件只接受固定版本CC BY/CC0，官方OAI和源码端点、4秒单连接限速/缓存、不执行TeX。许可记录不是对论文定理真伪的验证。

语料字段：paper_domain（源稿主题）、domain（formula上下文主题+显式matrix/PDE规则）、source_id/version/date/authors/title/source/file/line、license、raw_latex与转换、origin依据、semantic AST/CST/fallback、固定simple/complex及原始分量。数学知识难度未因来自研究论文就宣称每个短式都是高难；题材域也不等于严格数学类型。

拒收：未知参数宏/多行系统未专门解析、非数学prose/metadata、不平衡花括号、可疑\\数字控制序列/控制码、超长正文片段、exact duplicate、cross-partition duplicate、test词法模板在旧release或newfit中出现。安全symbol aliases展开保留原文；禁止计算TeX或附件指令。

benchmark质量是automated review+来源追溯，不是专家逐条认证。新test由来源/hash固定，算法训练前封存；发布主benchmark应再加数学专家抽样、双人标签核验和真正语义近重复排除，不能在这些未完成时标为gold semantic benchmark。

旧v2 release保留原hash，包含已知限制；v5包含旧集合与新增来源，新增质量门槛未静默修订旧记录。新test论文完整隔离并通过shape去重；极端目标域的specialized训练从文档和arXiv cross-list categories层面排除。缺样本类别空文件/统计明确，不用补造自然人工来源。

合成stress有parent_id和转换类型；它不是新的自然论文样本。去空白、重命名只是表示扰动，部分TeX上下文语义可能改变，损坏TeX仅用于byte losslessness/OOV速度，不评价数学正确性。

持久化test不能保证未来任何更新都“未见”。当前v5首次评估前未加载test文本；将来再以v5指标优化算法，应新建额外未触碰测试集或明确为development benchmark。

本次扩充实际下载119篇、117篇通过公式筛选，贡献25511条新公式；合计53858条。新test4375条来自30篇新增论文；极端域test1636条来自14篇；synthetic stress693条单列。old98+new119共217篇arXiv源码快照，不等于全部论文都有效公式大量贡献。CC BY4.0/3.0/CC0许可计数见source manifest，全部固定最新pre2020版本。

主corpus以无损gzip保存到GitHub（原120MiB压缩约15MiB），首次脚本自动解压并核验相同SHA；不是对LaTeX正规化或数据减少。formula_only、source registry、quality gates与全部split文件同release发布。
