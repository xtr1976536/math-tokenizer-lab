# AST历史复核与Cmd/Cross指标解释

复核日期：2026-10-15。读取并重新编码冻结checkpoint，不重新训练或修改模型。五个AST/词表控制×两容量×两场景，共20配置。全部token计数、Cmd、Num、Cross、paper-macroT/B与历史一致；独立Cross实现为区间布尔定义，不使用原sweep函数。新七轮计时另存，不替换原实验时间。

## 主测试比较

同4294式/27篇论文；相同指标实现。旧AST-v4自身训练词表不同，完整系统比较；same-dictionary硬AST、两级AST、Math-atomBPE、BoundaryAST词表完全一致。

|容量|方法|T/B↓|Cross%↓|Cmd%↑|原编码MiB/s↑|
|---|---|---:|---:|---:|---:|
|1024|历史AST-v4|0.41814|7.31|97.84|31.50|
|1024|同词表硬AST|0.47078|3.73|99.76|30.89|
|1024|BoundaryAST|0.39002|18.43|99.76|29.68|
|2048|历史AST-v4|0.35936|10.14|97.84|31.23|
|2048|同词表硬AST|0.41915|5.26|99.76|31.69|
|2048|BoundaryAST|0.32959|24.17|99.76|28.65|

BoundaryAST压缩更强；硬AST的结构约束更强，KaTeXcrossing更低。不能宣称BoundaryAST各项全面胜出。原位对照不是外部公认的唯一标准AST-BPE实现。

## 2048测试的比例分母

|方法|全部tokens|crossing tokens|Cross%|完整命令/命令总数|
|---|---:|---:|---:|---:|
|ast_boundary|106363|25710|24.17|30441/30513|
|official_atomic_bpe|108909|27086|24.87|30441/30513|
|Qwen2.5-0.5B|177232|38322|21.62|660/30513|
|meta-llama/Meta-Llama-3-8B|176675|38410|21.74|660/30513|
|tiktoken_o200k_base|174751|38873|22.24|668/30513|

Cross%同时受片段长度和结构跨越影响；同原文较小token容易在每个结构内，字节切分甚至可使此率接近0。绝对crossing次数也不是数学语义错误数。独立评估KaTeX与native成本的跨度来源也不同。

真实式：`x>2\sqrt{2}`。BoundaryAST切为`x | > | 2 | \sqrt{ | 2}`，命令完整但有一个crossing；千问切为`x | > | 2 | \ | sqrt | { | 2 | }`，命令被分开但crossing为0。

## 实验与术语

1024/2048是全部词表ID预算，包括256字节fallback、原子和BPE新增项；不是公式长度或模型参数量。成熟通用tokenizer保留完整原词表，因此两容量列复用同一通用模型。

主比较：broad fit30100+辅助512、dev965；新论文测试4294。来源域比较：specialized fit8124+辅助512、dev249；目标领域外1428式。均为tokenizer内在评估，不含下游LLM训练。

encode将原字符串变为ID序列；decode将ID/payload拼回原字符串。LLM生成的是ID，最终需解码以得到文本。速度表MiB/s指encode；decode另测。

FitBPB：fit频数加.5平滑的unigram概率，计算测试token的负log2概率总和/原UTF8字节。不是LM困惑度；大词表、计数稀疏和不同分词率均影响读数。

Used%：测试中实际出现ID种类数/全部ID空间；受词表规模和测试范围影响，不能仅凭低使用率判差。

详细checkpoint哈希、等词表校验、当前计时、逐式示例及独立计数见replay.json。
