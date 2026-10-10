# 独立数学分词器：首轮本地实验

这是 tokenizer-only pilot。奖励是训练内部留出数据的压缩增益，未训练或拼接任何语言模型，不能推断下游性能。

训练公式 1026 条，测试 406 条，总词表预算 384；自研方法随机种子 [17, 29, 43]。
所有方法使用同一测试 ID、原始 LaTeX、无截断、无特殊 token，逐条检查无损解码。AST 不支持的公式保留在共同测试集中，自研结构方法回退为 UTF-8 字节。

| 方法 | 比较组 | 实际词表 | token/公式（种子均值） | bytes/token | 无损率 | 编码 MiB/s | 训练秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| byte_bpe | trained | 384 | 14.377 | 1.713 | 100.0000% | 1.716 | 0.790 |
| ast_bpe | trained | 384 | 17.537 | 1.404 | 100.0000% | 0.809 | 1.426 |
| rl_bpe | trained | 384 | 14.479 | 1.701 | 100.0000% | 1.698 | 1.317 |
| rl_ast_bpe | trained | 384 | 17.644 | 1.395 | 100.0000% | 0.829 | 2.015 |
| byte_bpe_rl_fit_control | trained | 384 | 14.426 | 1.707 | 100.0000% | 1.705 | 0.653 |
| ast_bpe_rl_fit_control | trained | 384 | 17.555 | 1.402 | 100.0000% | 0.810 | 1.146 |
| utf8_bytes | reference | 256 | 24.621 | 1.000 | 100.0000% | 143.985 | 0.000 |
| hf_byte_bpe | trained | 384 | 14.377 | 1.713 | 100.0000% | 6.591 | 0.020 |
| sentencepiece_bpe | trained | 384 | 17.631 | 1.396 | 100.0000% | 11.203 | 0.012 |
| sentencepiece_unigram | trained | 384 | 17.606 | 1.398 | 100.0000% | 16.115 | 0.076 |
| tiktoken_gpt2 | pretrained_reference | 50257 | 14.788 | 1.665 | 100.0000% | 9.157 | 0.000 |
| tiktoken_cl100k_base | pretrained_reference | 100277 | 12.300 | 2.002 | 100.0000% | 10.105 | 0.000 |
| tiktoken_o200k_base | pretrained_reference | 200019 | 12.310 | 2.000 | 100.0000% | 13.977 | 0.000 |
| qwen2.5_official | pretrained_reference | 151665 | 12.345 | 1.994 | 100.0000% | 2.701 | 0.000 |

实际词表大小必须与预算同时解读：SentencePiece 需要未知符号和字节回退项；预训练参考具有不同规模词表及外部训练语料，不能用这张表宣称其训练方法优于或弱于自研方法。
RL 使用 80% 训练公式收集合并候选、20% 训练公式学习奖励；额外列出的 *_rl_fit_control 使用相同 80% 候选拟合子集，供判断策略学习效果。全部训练数据算法比较与内部消融比较都保留。

## 分领域／来源结果（自研种子均值和参考单次结果）

| 方法 | 领域 | 来源 | 公式数 | token/公式 | 无损率 |
|---|---|---|---:|---:|---:|
| ast_bpe | analysis | possible_mixed | 114 | 19.009 | 100.0000% |
| ast_bpe | analysis | unknown | 16 | 17.562 | 100.0000% |
| ast_bpe | combinatorics | possible_mixed | 120 | 15.875 | 100.0000% |
| ast_bpe | linear_algebra | ai_generated | 4 | 17.000 | 100.0000% |
| ast_bpe | logic | possible_mixed | 120 | 18.475 | 100.0000% |
| ast_bpe | pde | human | 26 | 9.385 | 100.0000% |
| ast_bpe | pde | possible_mixed | 6 | 39.667 | 100.0000% |
| ast_bpe_rl_fit_control | analysis | possible_mixed | 114 | 19.029 | 100.0000% |
| ast_bpe_rl_fit_control | analysis | unknown | 16 | 17.500 | 100.0000% |
| ast_bpe_rl_fit_control | combinatorics | possible_mixed | 120 | 15.839 | 100.0000% |
| ast_bpe_rl_fit_control | linear_algebra | ai_generated | 4 | 17.667 | 100.0000% |
| ast_bpe_rl_fit_control | logic | possible_mixed | 120 | 18.500 | 100.0000% |
| ast_bpe_rl_fit_control | pde | human | 26 | 9.538 | 100.0000% |
| ast_bpe_rl_fit_control | pde | possible_mixed | 6 | 39.778 | 100.0000% |
| byte_bpe | analysis | possible_mixed | 114 | 15.044 | 100.0000% |
| byte_bpe | analysis | unknown | 16 | 13.250 | 100.0000% |
| byte_bpe | combinatorics | possible_mixed | 120 | 13.017 | 100.0000% |
| byte_bpe | linear_algebra | ai_generated | 4 | 17.000 | 100.0000% |
| byte_bpe | logic | possible_mixed | 120 | 14.642 | 100.0000% |
| byte_bpe | pde | human | 26 | 12.038 | 100.0000% |
| byte_bpe | pde | possible_mixed | 6 | 35.000 | 100.0000% |
| byte_bpe_rl_fit_control | analysis | possible_mixed | 114 | 15.088 | 100.0000% |
| byte_bpe_rl_fit_control | analysis | unknown | 16 | 13.542 | 100.0000% |
| byte_bpe_rl_fit_control | combinatorics | possible_mixed | 120 | 13.097 | 100.0000% |
| byte_bpe_rl_fit_control | linear_algebra | ai_generated | 4 | 17.667 | 100.0000% |
| byte_bpe_rl_fit_control | logic | possible_mixed | 120 | 14.544 | 100.0000% |
| byte_bpe_rl_fit_control | pde | human | 26 | 12.359 | 100.0000% |
| byte_bpe_rl_fit_control | pde | possible_mixed | 6 | 35.222 | 100.0000% |
| hf_byte_bpe | analysis | possible_mixed | 114 | 15.044 | 100.0000% |
| hf_byte_bpe | analysis | unknown | 16 | 13.250 | 100.0000% |
| hf_byte_bpe | combinatorics | possible_mixed | 120 | 13.017 | 100.0000% |
| hf_byte_bpe | linear_algebra | ai_generated | 4 | 17.000 | 100.0000% |
| hf_byte_bpe | logic | possible_mixed | 120 | 14.642 | 100.0000% |
| hf_byte_bpe | pde | human | 26 | 12.038 | 100.0000% |
| hf_byte_bpe | pde | possible_mixed | 6 | 35.000 | 100.0000% |
| qwen2.5_official | analysis | possible_mixed | 114 | 11.623 | 100.0000% |
| qwen2.5_official | analysis | unknown | 16 | 11.688 | 100.0000% |
| qwen2.5_official | combinatorics | possible_mixed | 120 | 9.317 | 100.0000% |
| qwen2.5_official | linear_algebra | ai_generated | 4 | 11.000 | 100.0000% |
| qwen2.5_official | logic | possible_mixed | 120 | 9.150 | 100.0000% |
| qwen2.5_official | pde | human | 26 | 42.308 | 100.0000% |
| qwen2.5_official | pde | possible_mixed | 6 | 23.333 | 100.0000% |
| rl_ast_bpe | analysis | possible_mixed | 114 | 19.091 | 100.0000% |
| rl_ast_bpe | analysis | unknown | 16 | 17.604 | 100.0000% |
| rl_ast_bpe | combinatorics | possible_mixed | 120 | 15.969 | 100.0000% |
| rl_ast_bpe | linear_algebra | ai_generated | 4 | 19.000 | 100.0000% |
| rl_ast_bpe | logic | possible_mixed | 120 | 18.450 | 100.0000% |
| rl_ast_bpe | pde | human | 26 | 9.987 | 100.0000% |
| rl_ast_bpe | pde | possible_mixed | 6 | 39.889 | 100.0000% |
| rl_bpe | analysis | possible_mixed | 114 | 15.161 | 100.0000% |
| rl_bpe | analysis | unknown | 16 | 14.104 | 100.0000% |
| rl_bpe | combinatorics | possible_mixed | 120 | 13.186 | 100.0000% |
| rl_bpe | linear_algebra | ai_generated | 4 | 17.667 | 100.0000% |
| rl_bpe | logic | possible_mixed | 120 | 14.550 | 100.0000% |
| rl_bpe | pde | human | 26 | 12.346 | 100.0000% |
| rl_bpe | pde | possible_mixed | 6 | 34.056 | 100.0000% |
| sentencepiece_bpe | analysis | possible_mixed | 114 | 17.430 | 100.0000% |
| sentencepiece_bpe | analysis | unknown | 16 | 15.938 | 100.0000% |
| sentencepiece_bpe | combinatorics | possible_mixed | 120 | 15.192 | 100.0000% |
| sentencepiece_bpe | linear_algebra | ai_generated | 4 | 21.000 | 100.0000% |
| sentencepiece_bpe | logic | possible_mixed | 120 | 16.725 | 100.0000% |
| sentencepiece_bpe | pde | human | 26 | 29.846 | 100.0000% |
| sentencepiece_bpe | pde | possible_mixed | 6 | 37.667 | 100.0000% |
| sentencepiece_unigram | analysis | possible_mixed | 114 | 17.325 | 100.0000% |
| sentencepiece_unigram | analysis | unknown | 16 | 15.875 | 100.0000% |
| sentencepiece_unigram | combinatorics | possible_mixed | 120 | 14.683 | 100.0000% |
| sentencepiece_unigram | linear_algebra | ai_generated | 4 | 21.000 | 100.0000% |
| sentencepiece_unigram | logic | possible_mixed | 120 | 17.367 | 100.0000% |
| sentencepiece_unigram | pde | human | 26 | 29.115 | 100.0000% |
| sentencepiece_unigram | pde | possible_mixed | 6 | 38.667 | 100.0000% |
| tiktoken_cl100k_base | analysis | possible_mixed | 114 | 11.509 | 100.0000% |
| tiktoken_cl100k_base | analysis | unknown | 16 | 11.688 | 100.0000% |
| tiktoken_cl100k_base | combinatorics | possible_mixed | 120 | 9.275 | 100.0000% |
| tiktoken_cl100k_base | linear_algebra | ai_generated | 4 | 11.000 | 100.0000% |
| tiktoken_cl100k_base | logic | possible_mixed | 120 | 9.150 | 100.0000% |
| tiktoken_cl100k_base | pde | human | 26 | 42.308 | 100.0000% |
| tiktoken_cl100k_base | pde | possible_mixed | 6 | 23.333 | 100.0000% |
| tiktoken_gpt2 | analysis | possible_mixed | 114 | 13.868 | 100.0000% |
| tiktoken_gpt2 | analysis | unknown | 16 | 13.062 | 100.0000% |
| tiktoken_gpt2 | combinatorics | possible_mixed | 120 | 11.742 | 100.0000% |
| tiktoken_gpt2 | linear_algebra | ai_generated | 4 | 13.000 | 100.0000% |
| tiktoken_gpt2 | logic | possible_mixed | 120 | 12.108 | 100.0000% |
| tiktoken_gpt2 | pde | human | 26 | 43.269 | 100.0000% |
| tiktoken_gpt2 | pde | possible_mixed | 6 | 29.167 | 100.0000% |
| tiktoken_o200k_base | analysis | possible_mixed | 114 | 11.386 | 100.0000% |
| tiktoken_o200k_base | analysis | unknown | 16 | 11.438 | 100.0000% |
| tiktoken_o200k_base | combinatorics | possible_mixed | 120 | 9.208 | 100.0000% |
| tiktoken_o200k_base | linear_algebra | ai_generated | 4 | 11.000 | 100.0000% |
| tiktoken_o200k_base | logic | possible_mixed | 120 | 9.275 | 100.0000% |
| tiktoken_o200k_base | pde | human | 26 | 43.038 | 100.0000% |
| tiktoken_o200k_base | pde | possible_mixed | 6 | 22.667 | 100.0000% |
| utf8_bytes | analysis | possible_mixed | 114 | 19.623 | 100.0000% |
| utf8_bytes | analysis | unknown | 16 | 18.812 | 100.0000% |
| utf8_bytes | combinatorics | possible_mixed | 120 | 16.408 | 100.0000% |
| utf8_bytes | linear_algebra | ai_generated | 4 | 24.000 | 100.0000% |
| utf8_bytes | logic | possible_mixed | 120 | 18.692 | 100.0000% |
| utf8_bytes | pde | human | 26 | 111.231 | 100.0000% |
| utf8_bytes | pde | possible_mixed | 6 | 43.000 | 100.0000% |

## 配对组 bootstrap

按文档／公式家族重采样 1000 次；差值为 A-B 的 token/公式，负值有利于 A。这是给定训练种子的条件区间，不是训练随机性区间。

| 比较 | 差值 | 95% 区间 | 独立组数 |
|---|---:|---|---:|
| ast_bpe-vs-byte_bpe/seed-17 | 3.160 | 0.840, 3.631 | 32 |
| rl_bpe-vs-byte_bpe/seed-17 | 0.261 | 0.122, 0.630 | 32 |
| rl_ast_bpe-vs-byte_bpe/seed-17 | 3.271 | 1.350, 3.712 | 32 |
| rl_bpe-vs-fit-control/seed-17 | 0.192 | 0.071, 0.399 | 32 |
| rl_ast_bpe-vs-fit-control/seed-17 | 0.108 | 0.007, 0.476 | 32 |
| ast_bpe-vs-byte_bpe/seed-29 | 3.160 | 0.840, 3.631 | 32 |
| rl_bpe-vs-byte_bpe/seed-29 | -0.047 | -0.180, 0.333 | 32 |
| rl_ast_bpe-vs-byte_bpe/seed-29 | 3.268 | 1.498, 3.721 | 32 |
| rl_bpe-vs-fit-control/seed-29 | -0.074 | -0.119, 0.182 | 32 |
| rl_ast_bpe-vs-fit-control/seed-29 | 0.042 | -0.015, 0.309 | 32 |
| ast_bpe-vs-byte_bpe/seed-43 | 3.160 | 0.840, 3.631 | 32 |
| rl_bpe-vs-byte_bpe/seed-43 | 0.091 | -0.078, 0.345 | 32 |
| rl_ast_bpe-vs-byte_bpe/seed-43 | 3.261 | 1.174, 3.682 | 32 |
| rl_bpe-vs-fit-control/seed-43 | 0.039 | -0.131, 0.195 | 32 |
| rl_ast_bpe-vs-fit-control/seed-43 | 0.116 | -0.000, 0.344 | 32 |

## 解释与限制

1. 压缩、无损、解析覆盖率和速度是独立分词器指标，不等同于语言模型 BPB、数学推理或 PDE 识别准确率。
2. AST 是保留源位置的受限语义表达式树。未知宏、矩阵、求和／积分绑定和复杂系统尚未完整解析；回退比例可能主导压缩结果，必须先提高覆盖率再做方法结论。
3. RL 是离线 contextual merge policy，用 REINFORCE 学习候选合并选择。它不是 Dauncey & Wattenhofer 的端到端语言模型边界学习复现，也未实现其 early-exit 语言模型奖励。
4. human 表示原始数学结构的人工作者证据，LaTeX 的自动映射另有标签；unknown 不能当成人工，OpenAI 论文保守标为 possible_mixed。AI synthetic 是本助手编写模板的机械扩展，不能当作独立模型大样本。
5. 相同词表预算、相同测试协议不意味着计算预算相同；速度包括真实编码及结构解析，Python 和 Rust/C++ 的差异不能归因于算法本身。
6. 同文档隔离导致某些领域尚无独立测试文档；没有测试数据的类别不提供结果。分组 bootstrap 中部分组只有少量数据，置信区间不能外推到未收集领域。

原始逐公式结果、运行环境、输入 ID、数据／模型哈希与配置保存在本目录，可独立复核。
