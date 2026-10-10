# Tokenizer-only 指标协议 v2

所有指标使用同一公式 ID、同一未经模型特定正规化的原始 LaTeX 字节串，完整编码，无特殊 token、无截断。训练组预算含256字节回退；固定预训练组单列。统计量按微平均和文档宏平均区别报告；复杂度分类使用冻结公式规则，不由某个分词器决定。

| 指标 | 定义／用途 | 方向或限制 |
|---|---|---|
| token/公式 | 总token/公式数，另列文档宏平均 | 短并不等于下游好 |
| bytes/token、chars/token | 原UTF8字节／字符数除token数 | 压缩描述，字节单位不混用字符 |
| token/LaTeX词法单元 | 总token/共同词法单元数，公式fertility | 数学没有自然语言word，明确单位 |
| byte compression rate | 1-token数/原字节数 | 固定token计数压缩比例，不是磁盘压缩 |
| token长度p50/p90/p95/p99 | 每公式token数分位数 | 检查长公式尾部 |
| 平均token字节、p95 | 源payload字节长度 | 描述统计 |
| 实际/使用词表及利用率 | observed IDs / declared vocab | 依语料量，固定大词表与小词表谨慎比较 |
| type-token ratio | 观察到的ID种数/使用次数 | 不单向排名 |
| singleton type fraction | 一次使用ID数/观察ID数 | 小样本依赖 |
| byte token usage rate | 源payload长度1的token使用比例 | 低常对应更充分压缩 |
| Shannon entropy | 测试经验ID分布H1，bits/token | 描述统计，不是LM交叉熵 |
| Rényi效率 α=1,2,2.5,3 | Hα(P_ID)/log2(完整declared vocab) | 论文指标，不能替代全部指标 |
| 经验unigram bits/byte | H1*token数/原字节数 | 以测试自身分布估计的编码描述，不是LM BPB |
| fixed-width index bits/byte | ceil(log2(vocab))*token数/字节数 | 不含词表文件成本；字节与token计数关系 |
| bigram/trigram entropy | 官方TokEval条件熵，最小上下文计数3 | 原定义公开，低覆盖必须报告 |
| 无损还原率 | decode(encode(text))==text | 必须100%，失败不能删除 |
| UTF8 token完整率 | 单token payload可独立UTF8解码比例 | byte回退单token可能不完整，整体仍可无损 |
| lexeme preservation | 无token边界落入共同词法单元内部的比例 | 数字、命令拆分风险 |
| LaTeX command integrity | 无token边界落入反斜杠命令内部 | 不把宏碎片当完整命令 |
| number integrity | 数字串内部没有token边界比例 | 与三位数分组目标不同，分别报告 |
| operator isolation | 每个共同运算符的两侧都是token边界 | 操作符独立性，可能增加token数 |
| three-digit boundary F1 | 长整数从右每3位的参考边界与预测边界F1 | 仅>=4位整数；报告可评估样本数 |
| semantic AST覆盖率 | v1受限语义parser成功数 | 语义正确性仍需黄金标注 |
| LaTeX syntax tree覆盖率 | 语义失败后分组／命令叶子解析成功数 | 是CST，不能宣称语义AST |
| tree crossing rate | 不满足区间层状兼容的token数/可测token数 | 纯byte也可能为零，需联合压缩 |
| unseen/rare train token rate | 测试使用的ID在同tokenizer训练编码中0次或<=5次 | 外部预训练组只表示本语料频率，不是预训练稀有度 |
| 编码速度和训练秒数 | 三次清结构缓存后的完整编码中位数 | Rust/Python实现差异单独解释 |
| 固定分层结果 | 来源×领域×simple/complex | 来源与领域混杂，不能因果归因 |
| 组配对bootstrap | 按独立文档／结构家族重采样1000次 | 与训练种子变异分开 |

## 文献与复用

- [TokEval: A Tokenizer Evaluation Suite, 2026](https://arxiv.org/abs/2608.18062)：多指标intrinsic评估；[官方源码](https://github.com/cimeister/tokenizer-intrinsic-evals)固定commit `7d03a61c2e2b4b557b796519141ccb52d144e10f`。本项目直接调用其Rényi、bigram、trigram函数，未改官方文件；通过测试交叉核对自算Rényi。领域限定的命令完整性等属于本项目自定义扩展。
- [Tokenization and the Noiseless Channel, ACL 2023](https://aclanthology.org/2023.acl-long.284/)：Rényi效率，按声明词表大小正规化。
- [Two Counterexamples to Tokenization and the Noiseless Channel, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.1469/)：Rényi变好不保证下游变好，避免单指标论证。
- [Tokenizer Choice For LLM Training: Negligible or Crucial?, 2023](https://arxiv.org/abs/2310.08754)：fertility与parity不能独立可靠预测全部下游。
- [Subword Regularization, ACL 2018](https://aclanthology.org/P18-1007/)：同词表多分词路径及Unigram公开基准。
- [BPE-Dropout, ACL 2020](https://aclanthology.org/2020.acl-main.170/)：固定BPE词表随机路径思想，不是RL。当前正式推断必须确定性，未用随机测试的最好结果。
- [Tokenization with Split Trees, 2026](https://arxiv.org/abs/2605.22705)：词表选择／分词路径对压缩的重要性；统计split tree不是数学AST。未找到官方实现，不冒充已跑通基准。

不适用的指标：自然语言MorphScore、跨语言公平Gini、程序缩进等不能直接照搬成数学公式质量。下游模型未训练，因此不报告真实语言模型perplexity、BPB、生成正确性。
