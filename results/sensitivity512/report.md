# 独立数学分词器：首轮本地实验

这是 tokenizer-only pilot。奖励是训练内部留出数据的压缩增益，未训练或拼接任何语言模型，不能推断下游性能。

训练公式 1026 条，测试 406 条，总词表预算 512；自研方法随机种子 [17, 29, 43]。
所有方法使用同一测试 ID、原始 LaTeX、无截断、无特殊 token，逐条检查无损解码。AST 不支持的公式保留在共同测试集中，自研结构方法回退为 UTF-8 字节。

| 方法 | 比较组 | 实际词表 | token/公式（种子均值） | bytes/token | 无损率 | 编码 MiB/s | 训练秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| byte_bpe | trained | 512 | 12.616 | 1.952 | 100.0000% | 1.078 | 1.495 |
| ast_bpe | trained | 512 | 16.818 | 1.464 | 100.0000% | 0.709 | 2.279 |
| rl_bpe | trained | 512 | 12.532 | 1.965 | 100.0000% | 1.073 | 2.354 |
| rl_ast_bpe | trained | 512 | 16.843 | 1.462 | 100.0000% | 0.702 | 3.374 |
| byte_bpe_rl_fit_control | trained | 512 | 12.571 | 1.959 | 100.0000% | 1.082 | 1.222 |
| ast_bpe_rl_fit_control | trained | 512 | 16.814 | 1.464 | 100.0000% | 0.702 | 1.820 |
| utf8_bytes | reference | 256 | 24.621 | 1.000 | 100.0000% | 133.249 | 0.000 |
| hf_byte_bpe | trained | 512 | 12.616 | 1.952 | 100.0000% | 6.684 | 0.024 |
| sentencepiece_bpe | trained | 512 | 14.357 | 1.715 | 100.0000% | 11.240 | 0.014 |
| sentencepiece_unigram | trained | 512 | 14.204 | 1.733 | 100.0000% | 17.043 | 0.064 |
| tiktoken_gpt2 | pretrained_reference | 50257 | 14.788 | 1.665 | 100.0000% | 10.091 | 0.000 |
| tiktoken_cl100k_base | pretrained_reference | 100277 | 12.300 | 2.002 | 100.0000% | 9.941 | 0.000 |
| tiktoken_o200k_base | pretrained_reference | 200019 | 12.310 | 2.000 | 100.0000% | 13.605 | 0.000 |
| qwen2.5_official | pretrained_reference | 151665 | 12.345 | 1.994 | 100.0000% | 2.679 | 0.000 |

实际词表大小必须与预算同时解读：SentencePiece 需要未知符号和字节回退项；预训练参考具有不同规模词表及外部训练语料，不能用这张表宣称其训练方法优于或弱于自研方法。
RL 使用 80% 训练公式收集合并候选、20% 训练公式学习奖励；额外列出的 *_rl_fit_control 使用相同 80% 候选拟合子集，供判断策略学习效果。全部训练数据算法比较与内部消融比较都保留。

## 分领域／来源结果（自研种子均值和参考单次结果）

| 方法 | 领域 | 来源 | 公式数 | token/公式 | 无损率 |
|---|---|---|---:|---:|---:|
| ast_bpe | analysis | possible_mixed | 114 | 18.561 | 100.0000% |
| ast_bpe | analysis | unknown | 16 | 16.875 | 100.0000% |
| ast_bpe | combinatorics | possible_mixed | 120 | 15.092 | 100.0000% |
| ast_bpe | linear_algebra | ai_generated | 4 | 9.000 | 100.0000% |
| ast_bpe | logic | possible_mixed | 120 | 18.292 | 100.0000% |
| ast_bpe | pde | human | 26 | 6.577 | 100.0000% |
| ast_bpe | pde | possible_mixed | 6 | 38.167 | 100.0000% |
| ast_bpe_rl_fit_control | analysis | possible_mixed | 114 | 18.599 | 100.0000% |
| ast_bpe_rl_fit_control | analysis | unknown | 16 | 16.875 | 100.0000% |
| ast_bpe_rl_fit_control | combinatorics | possible_mixed | 120 | 15.081 | 100.0000% |
| ast_bpe_rl_fit_control | linear_algebra | ai_generated | 4 | 8.667 | 100.0000% |
| ast_bpe_rl_fit_control | logic | possible_mixed | 120 | 18.289 | 100.0000% |
| ast_bpe_rl_fit_control | pde | human | 26 | 6.474 | 100.0000% |
| ast_bpe_rl_fit_control | pde | possible_mixed | 6 | 38.167 | 100.0000% |
| byte_bpe | analysis | possible_mixed | 114 | 13.175 | 100.0000% |
| byte_bpe | analysis | unknown | 16 | 12.125 | 100.0000% |
| byte_bpe | combinatorics | possible_mixed | 120 | 11.550 | 100.0000% |
| byte_bpe | linear_algebra | ai_generated | 4 | 11.000 | 100.0000% |
| byte_bpe | logic | possible_mixed | 120 | 12.975 | 100.0000% |
| byte_bpe | pde | human | 26 | 9.269 | 100.0000% |
| byte_bpe | pde | possible_mixed | 6 | 33.000 | 100.0000% |
| byte_bpe_rl_fit_control | analysis | possible_mixed | 114 | 13.056 | 100.0000% |
| byte_bpe_rl_fit_control | analysis | unknown | 16 | 11.688 | 100.0000% |
| byte_bpe_rl_fit_control | combinatorics | possible_mixed | 120 | 11.531 | 100.0000% |
| byte_bpe_rl_fit_control | linear_algebra | ai_generated | 4 | 10.333 | 100.0000% |
| byte_bpe_rl_fit_control | logic | possible_mixed | 120 | 13.056 | 100.0000% |
| byte_bpe_rl_fit_control | pde | human | 26 | 9.295 | 100.0000% |
| byte_bpe_rl_fit_control | pde | possible_mixed | 6 | 32.500 | 100.0000% |
| hf_byte_bpe | analysis | possible_mixed | 114 | 13.175 | 100.0000% |
| hf_byte_bpe | analysis | unknown | 16 | 12.125 | 100.0000% |
| hf_byte_bpe | combinatorics | possible_mixed | 120 | 11.550 | 100.0000% |
| hf_byte_bpe | linear_algebra | ai_generated | 4 | 11.000 | 100.0000% |
| hf_byte_bpe | logic | possible_mixed | 120 | 12.975 | 100.0000% |
| hf_byte_bpe | pde | human | 26 | 9.269 | 100.0000% |
| hf_byte_bpe | pde | possible_mixed | 6 | 33.000 | 100.0000% |
| qwen2.5_official | analysis | possible_mixed | 114 | 11.623 | 100.0000% |
| qwen2.5_official | analysis | unknown | 16 | 11.688 | 100.0000% |
| qwen2.5_official | combinatorics | possible_mixed | 120 | 9.317 | 100.0000% |
| qwen2.5_official | linear_algebra | ai_generated | 4 | 11.000 | 100.0000% |
| qwen2.5_official | logic | possible_mixed | 120 | 9.150 | 100.0000% |
| qwen2.5_official | pde | human | 26 | 42.308 | 100.0000% |
| qwen2.5_official | pde | possible_mixed | 6 | 23.333 | 100.0000% |
| rl_ast_bpe | analysis | possible_mixed | 114 | 18.623 | 100.0000% |
| rl_ast_bpe | analysis | unknown | 16 | 16.854 | 100.0000% |
| rl_ast_bpe | combinatorics | possible_mixed | 120 | 15.164 | 100.0000% |
| rl_ast_bpe | linear_algebra | ai_generated | 4 | 8.667 | 100.0000% |
| rl_ast_bpe | logic | possible_mixed | 120 | 18.242 | 100.0000% |
| rl_ast_bpe | pde | human | 26 | 6.603 | 100.0000% |
| rl_ast_bpe | pde | possible_mixed | 6 | 38.444 | 100.0000% |
| rl_bpe | analysis | possible_mixed | 114 | 13.003 | 100.0000% |
| rl_bpe | analysis | unknown | 16 | 11.708 | 100.0000% |
| rl_bpe | combinatorics | possible_mixed | 120 | 11.583 | 100.0000% |
| rl_bpe | linear_algebra | ai_generated | 4 | 10.333 | 100.0000% |
| rl_bpe | logic | possible_mixed | 120 | 12.969 | 100.0000% |
| rl_bpe | pde | human | 26 | 9.321 | 100.0000% |
| rl_bpe | pde | possible_mixed | 6 | 31.389 | 100.0000% |
| sentencepiece_bpe | analysis | possible_mixed | 114 | 13.675 | 100.0000% |
| sentencepiece_bpe | analysis | unknown | 16 | 13.688 | 100.0000% |
| sentencepiece_bpe | combinatorics | possible_mixed | 120 | 12.375 | 100.0000% |
| sentencepiece_bpe | linear_algebra | ai_generated | 4 | 11.000 | 100.0000% |
| sentencepiece_bpe | logic | possible_mixed | 120 | 13.758 | 100.0000% |
| sentencepiece_bpe | pde | human | 26 | 26.500 | 100.0000% |
| sentencepiece_bpe | pde | possible_mixed | 6 | 30.333 | 100.0000% |
| sentencepiece_unigram | analysis | possible_mixed | 114 | 13.342 | 100.0000% |
| sentencepiece_unigram | analysis | unknown | 16 | 12.938 | 100.0000% |
| sentencepiece_unigram | combinatorics | possible_mixed | 120 | 12.267 | 100.0000% |
| sentencepiece_unigram | linear_algebra | ai_generated | 4 | 9.750 | 100.0000% |
| sentencepiece_unigram | logic | possible_mixed | 120 | 13.917 | 100.0000% |
| sentencepiece_unigram | pde | human | 26 | 27.385 | 100.0000% |
| sentencepiece_unigram | pde | possible_mixed | 6 | 24.333 | 100.0000% |
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
| ast_bpe-vs-byte_bpe/seed-17 | 4.202 | 0.783, 4.905 | 32 |
| rl_bpe-vs-byte_bpe/seed-17 | -0.032 | -0.228, 0.142 | 32 |
| rl_ast_bpe-vs-byte_bpe/seed-17 | 4.268 | 0.866, 4.969 | 32 |
| rl_bpe-vs-fit-control/seed-17 | 0.027 | -0.047, 0.073 | 32 |
| rl_ast_bpe-vs-fit-control/seed-17 | 0.067 | 0.007, 0.166 | 32 |
| ast_bpe-vs-byte_bpe/seed-29 | 4.202 | 0.783, 4.905 | 32 |
| rl_bpe-vs-byte_bpe/seed-29 | -0.128 | -0.303, 0.120 | 32 |
| rl_ast_bpe-vs-byte_bpe/seed-29 | 4.204 | 0.690, 4.863 | 32 |
| rl_bpe-vs-fit-control/seed-29 | -0.135 | -0.305, 0.173 | 32 |
| rl_ast_bpe-vs-fit-control/seed-29 | -0.034 | -0.140, 0.083 | 32 |
| ast_bpe-vs-byte_bpe/seed-43 | 4.202 | 0.783, 4.905 | 32 |
| rl_bpe-vs-byte_bpe/seed-43 | -0.091 | -0.436, -0.014 | 32 |
| rl_ast_bpe-vs-byte_bpe/seed-43 | 4.209 | 0.603, 4.948 | 32 |
| rl_bpe-vs-fit-control/seed-43 | -0.007 | -0.117, 0.104 | 32 |
| rl_ast_bpe-vs-fit-control/seed-43 | 0.054 | -0.023, 0.143 | 32 |

## 解释与限制

1. 压缩、无损、解析覆盖率和速度是独立分词器指标，不等同于语言模型 BPB、数学推理或 PDE 识别准确率。
2. AST 是保留源位置的受限语义表达式树。未知宏、矩阵、求和／积分绑定和复杂系统尚未完整解析；回退比例可能主导压缩结果，必须先提高覆盖率再做方法结论。
3. RL 是离线 contextual merge policy，用 REINFORCE 学习候选合并选择。它不是 Dauncey & Wattenhofer 的端到端语言模型边界学习复现，也未实现其 early-exit 语言模型奖励。
4. human 表示原始数学结构的人工作者证据，LaTeX 的自动映射另有标签；unknown 不能当成人工，OpenAI 论文保守标为 possible_mixed。AI synthetic 是本助手编写模板的机械扩展，不能当作独立模型大样本。
5. 相同词表预算、相同测试协议不意味着计算预算相同；速度包括真实编码及结构解析，Python 和 Rust/C++ 的差异不能归因于算法本身。
6. 同文档隔离导致某些领域尚无独立测试文档；没有测试数据的类别不提供结果。分组 bootstrap 中部分组只有少量数据，置信区间不能外推到未收集领域。

原始逐公式结果、运行环境、输入 ID、数据／模型哈希与配置保存在本目录，可独立复核。
