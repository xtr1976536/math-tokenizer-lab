# 独立数学分词器：首轮本地实验

这是 tokenizer-only pilot。奖励是训练内部留出数据的压缩增益，未训练或拼接任何语言模型，不能推断下游性能。

训练公式 80 条，测试 60 条，总词表预算 384；自研方法随机种子 [17]。
所有方法使用同一测试 ID、原始 LaTeX、无截断、无特殊 token，逐条检查无损解码。AST 不支持的公式保留在共同测试集中，自研结构方法回退为 UTF-8 字节。

| 方法 | 比较组 | 实际词表 | token/公式（种子均值） | bytes/token | 无损率 | 编码 MiB/s | 训练秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| byte_bpe | trained | 384 | 11.750 | 1.810 | 100.0000% | 1.645 | 0.056 |
| ast_bpe | trained | 384 | 14.200 | 1.498 | 100.0000% | 0.781 | 0.078 |
| rl_bpe | trained | 384 | 12.233 | 1.738 | 100.0000% | 1.630 | 0.133 |
| rl_ast_bpe | trained | 384 | 14.517 | 1.465 | 100.0000% | 0.792 | 0.154 |
| byte_bpe_rl_fit_control | trained | 384 | 12.133 | 1.753 | 100.0000% | 1.632 | 0.044 |
| ast_bpe_rl_fit_control | trained | 384 | 14.533 | 1.463 | 100.0000% | 0.792 | 0.053 |
| utf8_bytes | reference | 256 | 21.267 | 1.000 | 100.0000% | 152.895 | 0.000 |
| hf_byte_bpe | trained | 384 | 11.800 | 1.802 | 100.0000% | 6.957 | 0.005 |
| sentencepiece_bpe | trained | 384 | 14.250 | 1.492 | 100.0000% | 12.561 | 0.003 |
| sentencepiece_unigram | trained | 384 | 14.317 | 1.485 | 100.0000% | 16.575 | 0.004 |
| tiktoken_gpt2 | pretrained_reference | 50257 | 12.683 | 1.677 | 100.0000% | 8.602 | 0.000 |
| tiktoken_cl100k_base | pretrained_reference | 100277 | 10.667 | 1.994 | 100.0000% | 8.858 | 0.000 |
| tiktoken_o200k_base | pretrained_reference | 200019 | 10.733 | 1.981 | 100.0000% | 15.085 | 0.000 |
| qwen2.5_official | pretrained_reference | 151665 | 10.700 | 1.988 | 100.0000% | 2.513 | 0.000 |

实际词表大小必须与预算同时解读：SentencePiece 需要未知符号和字节回退项；预训练参考具有不同规模词表及外部训练语料，不能用这张表宣称其训练方法优于或弱于自研方法。
RL 使用 80% 训练公式收集合并候选、20% 训练公式学习奖励；额外列出的 *_rl_fit_control 使用相同 80% 候选拟合子集，供判断策略学习效果。全部训练数据算法比较与内部消融比较都保留。

## 分领域／来源结果（自研种子均值和参考单次结果）

| 方法 | 领域 | 来源 | 公式数 | token/公式 | 无损率 |
|---|---|---|---:|---:|---:|
| ast_bpe | analysis | possible_mixed | 11 | 7.545 | 100.0000% |
| ast_bpe | analysis | unknown | 1 | 12.000 | 100.0000% |
| ast_bpe | combinatorics | possible_mixed | 11 | 16.273 | 100.0000% |
| ast_bpe | logic | possible_mixed | 33 | 15.939 | 100.0000% |
| ast_bpe | pde | human | 4 | 13.000 | 100.0000% |
| ast_bpe_rl_fit_control | analysis | possible_mixed | 11 | 7.545 | 100.0000% |
| ast_bpe_rl_fit_control | analysis | unknown | 1 | 12.000 | 100.0000% |
| ast_bpe_rl_fit_control | combinatorics | possible_mixed | 11 | 16.091 | 100.0000% |
| ast_bpe_rl_fit_control | logic | possible_mixed | 33 | 15.970 | 100.0000% |
| ast_bpe_rl_fit_control | pde | human | 4 | 18.250 | 100.0000% |
| byte_bpe | analysis | possible_mixed | 11 | 6.636 | 100.0000% |
| byte_bpe | analysis | unknown | 1 | 10.000 | 100.0000% |
| byte_bpe | combinatorics | possible_mixed | 11 | 13.727 | 100.0000% |
| byte_bpe | logic | possible_mixed | 33 | 12.576 | 100.0000% |
| byte_bpe | pde | human | 4 | 14.000 | 100.0000% |
| byte_bpe_rl_fit_control | analysis | possible_mixed | 11 | 6.727 | 100.0000% |
| byte_bpe_rl_fit_control | analysis | unknown | 1 | 9.000 | 100.0000% |
| byte_bpe_rl_fit_control | combinatorics | possible_mixed | 11 | 13.727 | 100.0000% |
| byte_bpe_rl_fit_control | logic | possible_mixed | 33 | 12.667 | 100.0000% |
| byte_bpe_rl_fit_control | pde | human | 4 | 19.000 | 100.0000% |
| hf_byte_bpe | analysis | possible_mixed | 11 | 6.636 | 100.0000% |
| hf_byte_bpe | analysis | unknown | 1 | 9.000 | 100.0000% |
| hf_byte_bpe | combinatorics | possible_mixed | 11 | 13.727 | 100.0000% |
| hf_byte_bpe | logic | possible_mixed | 33 | 12.636 | 100.0000% |
| hf_byte_bpe | pde | human | 4 | 14.500 | 100.0000% |
| qwen2.5_official | analysis | possible_mixed | 11 | 5.091 | 100.0000% |
| qwen2.5_official | analysis | unknown | 1 | 10.000 | 100.0000% |
| qwen2.5_official | combinatorics | possible_mixed | 11 | 10.727 | 100.0000% |
| qwen2.5_official | logic | possible_mixed | 33 | 8.455 | 100.0000% |
| qwen2.5_official | pde | human | 4 | 44.750 | 100.0000% |
| rl_ast_bpe | analysis | possible_mixed | 11 | 7.364 | 100.0000% |
| rl_ast_bpe | analysis | unknown | 1 | 12.000 | 100.0000% |
| rl_ast_bpe | combinatorics | possible_mixed | 11 | 16.182 | 100.0000% |
| rl_ast_bpe | logic | possible_mixed | 33 | 15.970 | 100.0000% |
| rl_ast_bpe | pde | human | 4 | 18.250 | 100.0000% |
| rl_bpe | analysis | possible_mixed | 11 | 6.727 | 100.0000% |
| rl_bpe | analysis | unknown | 1 | 9.000 | 100.0000% |
| rl_bpe | combinatorics | possible_mixed | 11 | 13.909 | 100.0000% |
| rl_bpe | logic | possible_mixed | 33 | 12.788 | 100.0000% |
| rl_bpe | pde | human | 4 | 19.000 | 100.0000% |
| sentencepiece_bpe | analysis | possible_mixed | 11 | 7.727 | 100.0000% |
| sentencepiece_bpe | analysis | unknown | 1 | 13.000 | 100.0000% |
| sentencepiece_bpe | combinatorics | possible_mixed | 11 | 15.364 | 100.0000% |
| sentencepiece_bpe | logic | possible_mixed | 33 | 13.909 | 100.0000% |
| sentencepiece_bpe | pde | human | 4 | 32.250 | 100.0000% |
| sentencepiece_unigram | analysis | possible_mixed | 11 | 7.818 | 100.0000% |
| sentencepiece_unigram | analysis | unknown | 1 | 13.000 | 100.0000% |
| sentencepiece_unigram | combinatorics | possible_mixed | 11 | 15.091 | 100.0000% |
| sentencepiece_unigram | logic | possible_mixed | 33 | 14.152 | 100.0000% |
| sentencepiece_unigram | pde | human | 4 | 31.750 | 100.0000% |
| tiktoken_cl100k_base | analysis | possible_mixed | 11 | 5.000 | 100.0000% |
| tiktoken_cl100k_base | analysis | unknown | 1 | 10.000 | 100.0000% |
| tiktoken_cl100k_base | combinatorics | possible_mixed | 11 | 10.636 | 100.0000% |
| tiktoken_cl100k_base | logic | possible_mixed | 33 | 8.455 | 100.0000% |
| tiktoken_cl100k_base | pde | human | 4 | 44.750 | 100.0000% |
| tiktoken_gpt2 | analysis | possible_mixed | 11 | 6.455 | 100.0000% |
| tiktoken_gpt2 | analysis | unknown | 1 | 10.000 | 100.0000% |
| tiktoken_gpt2 | combinatorics | possible_mixed | 11 | 13.091 | 100.0000% |
| tiktoken_gpt2 | logic | possible_mixed | 33 | 10.727 | 100.0000% |
| tiktoken_gpt2 | pde | human | 4 | 45.500 | 100.0000% |
| tiktoken_o200k_base | analysis | possible_mixed | 11 | 5.000 | 100.0000% |
| tiktoken_o200k_base | analysis | unknown | 1 | 10.000 | 100.0000% |
| tiktoken_o200k_base | combinatorics | possible_mixed | 11 | 10.455 | 100.0000% |
| tiktoken_o200k_base | logic | possible_mixed | 33 | 8.545 | 100.0000% |
| tiktoken_o200k_base | pde | human | 4 | 45.500 | 100.0000% |
| utf8_bytes | analysis | possible_mixed | 11 | 8.000 | 100.0000% |
| utf8_bytes | analysis | unknown | 1 | 13.000 | 100.0000% |
| utf8_bytes | combinatorics | possible_mixed | 11 | 16.455 | 100.0000% |
| utf8_bytes | logic | possible_mixed | 33 | 16.061 | 100.0000% |
| utf8_bytes | pde | human | 4 | 116.000 | 100.0000% |

## 配对组 bootstrap

按文档／公式家族重采样 1000 次；差值为 A-B 的 token/公式，负值有利于 A。这是给定训练种子的条件区间，不是训练随机性区间。

| 比较 | 差值 | 95% 区间 | 独立组数 |
|---|---:|---|---:|
| ast_bpe-vs-byte_bpe/seed-17 | 2.450 | -0.126, 3.184 | 8 |
| rl_bpe-vs-byte_bpe/seed-17 | 0.483 | 0.205, 2.611 | 8 |
| rl_ast_bpe-vs-byte_bpe/seed-17 | 2.767 | 1.111, 3.800 | 8 |
| rl_bpe-vs-fit-control/seed-17 | 0.100 | 0.000, 0.143 | 8 |
| rl_ast_bpe-vs-fit-control/seed-17 | -0.017 | -0.143, 0.071 | 8 |

## 解释与限制

1. 压缩、无损、解析覆盖率和速度是独立分词器指标，不等同于语言模型 BPB、数学推理或 PDE 识别准确率。
2. AST 是保留源位置的受限语义表达式树。未知宏、矩阵、求和／积分绑定和复杂系统尚未完整解析；回退比例可能主导压缩结果，必须先提高覆盖率再做方法结论。
3. RL 是离线 contextual merge policy，用 REINFORCE 学习候选合并选择。它不是 Dauncey & Wattenhofer 的端到端语言模型边界学习复现，也未实现其 early-exit 语言模型奖励。
4. human 表示原始数学结构的人工作者证据，LaTeX 的自动映射另有标签；unknown 不能当成人工，OpenAI 论文保守标为 possible_mixed。AI synthetic 是本助手编写模板的机械扩展，不能当作独立模型大样本。
5. 相同词表预算、相同测试协议不意味着计算预算相同；速度包括真实编码及结构解析，Python 和 Rust/C++ 的差异不能归因于算法本身。
6. 同文档隔离导致某些领域尚无独立测试文档；没有测试数据的类别不提供结果。分组 bootstrap 中部分组只有少量数据，置信区间不能外推到未收集领域。

原始逐公式结果、运行环境、输入 ID、数据／模型哈希与配置保存在本目录，可独立复核。
