# 第二版独立数学分词器实验

训练80条、验证40条、测试40条；训练词表预算512；随机种子[17, 29, 43]。
RL完整路径奖励只使用训练公式；配置仅按验证文档宏平均token/byte选择，测试在selection.json冻结后统一调用。不同模型都用相同原始LaTeX，无截断、无特殊token。
结构树分 semantic_ast 与 latex_syntax_tree，后者是保留命令和嵌套的语法树，不能称通用语义AST。

| 方法 | 比较组 | token/公式 | bytes/token | 词表利用率 | Rényi效率 α=2 | 命令完整率 | 跨树比例 | 无损率 | 编码MiB/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| byte_bpe | trained | 19.6750 | 1.9136 | 0.3438 | 0.6819 | 0.4522 | 0.1842 | 1.0000 | 2.8993 |
| byte_shortest_path | trained | 19.6250 | 1.9185 | 0.3418 | 0.6695 | 0.4957 | 0.1694 | 1.0000 | 0.1651 |
| byte_policy_no_rl | trained | 19.7750 | 1.9039 | 0.3457 | 0.6870 | 0.4435 | 0.1820 | 1.0000 | 0.1018 |
| rl_bpe_v2 | trained | 19.7500 | 1.9063 | 0.3470 | 0.6875 | 0.4522 | 0.1806 | 1.0000 | 0.1020 |
| ast_bpe_v2 | trained | 21.7000 | 1.7350 | 0.3125 | 0.6104 | 0.5565 | 0.0000 | 1.0000 | 0.6306 |
| ast_shortest_path | trained | 21.7000 | 1.7350 | 0.3125 | 0.6081 | 0.5565 | 0.0000 | 1.0000 | 0.1270 |
| ast_policy_no_rl | trained | 21.7500 | 1.7310 | 0.3105 | 0.6075 | 0.5565 | 0.0000 | 1.0000 | 0.0862 |
| rl_ast_bpe_v2 | trained | 21.7500 | 1.7310 | 0.3105 | 0.6075 | 0.5565 | 0.0000 | 1.0000 | 0.0867 |
| ast_bpe_v1 | trained | 31.8000 | 1.1840 | 0.1953 | 0.5354 | 0.2522 | 0.0000 | 1.0000 | 1.1396 |
| rl_bpe_v1 | trained | 21.0250 | 1.7907 | 0.3359 | 0.6736 | 0.2696 | 0.1772 | 1.0000 | 1.3897 |
| rl_ast_bpe_v1 | trained | 32.5500 | 1.1567 | 0.1875 | 0.5290 | 0.2522 | 0.0000 | 1.0000 | 1.1194 |
| hf_byte_bpe | trained | 19.7000 | 1.9112 | 0.3418 | 0.6778 | 0.4522 | 0.1815 | 1.0000 | 7.7338 |
| sentencepiece_bpe | trained | 22.4000 | 1.6808 | 0.3203 | 0.6709 | 0.0000 | 0.1819 | 1.0000 | 13.8823 |
| sentencepiece_unigram | trained | 22.2750 | 1.6902 | 0.3086 | 0.6474 | 0.0000 | 0.1661 | 1.0000 | 22.6625 |
| utf8_bytes | reference | 37.6500 | 1.0000 | 0.2852 | 0.5817 | 0.0000 | 0.0000 | 1.0000 | 173.2071 |
| tiktoken_gpt2 | pretrained_reference | 21.3500 | 1.7635 | 0.0035 | 0.3573 | 0.0000 | 0.2260 | 1.0000 | 10.7818 |
| tiktoken_cl100k_base | pretrained_reference | 19.3750 | 1.9432 | 0.0020 | 0.3458 | 0.0000 | 0.2735 | 1.0000 | 10.3731 |
| tiktoken_o200k_base | pretrained_reference | 19.3000 | 1.9508 | 0.0010 | 0.3270 | 0.0000 | 0.2733 | 1.0000 | 14.9933 |
| qwen2.5_official | pretrained_reference | 19.8500 | 1.8967 | 0.0013 | 0.3296 | 0.0000 | 0.2670 | 1.0000 | 2.7296 |

## 验证集消融与选择

固定选择：{'byte': 'naive', 'ast': 'naive'}

| 分支 | 配置 | 两种子验证文档宏平均token/byte |
|---|---|---:|
| byte | naive | 0.594975 |
| byte | relative | 0.595808 |
| byte | rloo | 0.596641 |
| byte | rloo_discount | 0.596641 |
| byte | rloo_entropy | 0.596641 |
| byte | rloo_longer | 0.596641 |
| byte | rloo_soft_structure | 0.596641 |
| byte | rloo_zero_prior | 0.610586 |
| ast | naive | 0.626231 |
| ast | relative | 0.626231 |
| ast | rloo | 0.626231 |
| ast | rloo_discount | 0.626231 |
| ast | rloo_entropy | 0.626231 |
| ast | rloo_longer | 0.626231 |
| ast | rloo_zero_prior | 0.626231 |

所有配置、两个种子、epoch训练损失、梯度样本方差均保存在dev_ablations.json；没有按测试选择最好的配置。no_rl与shortest_path控制使用同一BPE词表：前者测试无策略学习时的长度先验，后者是这个固定词表与相同约束下的最少token路径参照。

## 按来源、领域、复杂度分层

| 方法 | 领域 | 来源 | 复杂度 | 独立公式数 | token/公式 | token/byte |
|---|---|---|---|---:|---:|---:|
| ast_bpe_v1 | algebra | human | complex | 1 | 143.0000 | 1.0000 |
| ast_bpe_v1 | algebra | human | simple | 12 | 14.6667 | 0.9514 |
| ast_bpe_v1 | analysis | unknown | simple | 2 | 69.0000 | 0.8961 |
| ast_bpe_v1 | combinatorics | possible_mixed | simple | 2 | 27.5000 | 0.9483 |
| ast_bpe_v1 | combinatorics | unknown | simple | 1 | 106.0000 | 1.0000 |
| ast_bpe_v1 | geometry_topology | human | simple | 10 | 19.5000 | 0.9112 |
| ast_bpe_v1 | number_theory | unknown | complex | 1 | 129.0000 | 1.0000 |
| ast_bpe_v1 | optimization | human | simple | 6 | 14.1667 | 0.8173 |
| ast_bpe_v1 | pde | human | complex | 1 | 77.0000 | 0.3929 |
| ast_bpe_v1 | pde | human | simple | 1 | 41.0000 | 0.4824 |
| ast_bpe_v1 | probability | human | simple | 3 | 42.3333 | 0.9621 |
| ast_bpe_v2 | algebra | human | complex | 1 | 112.0000 | 0.7832 |
| ast_bpe_v2 | algebra | human | simple | 12 | 11.7500 | 0.7622 |
| ast_bpe_v2 | analysis | unknown | simple | 2 | 33.0000 | 0.4286 |
| ast_bpe_v2 | combinatorics | possible_mixed | simple | 2 | 23.5000 | 0.8103 |
| ast_bpe_v2 | combinatorics | unknown | simple | 1 | 36.0000 | 0.3396 |
| ast_bpe_v2 | geometry_topology | human | simple | 10 | 13.8000 | 0.6449 |
| ast_bpe_v2 | number_theory | unknown | complex | 1 | 57.0000 | 0.4419 |
| ast_bpe_v2 | optimization | human | simple | 6 | 9.3333 | 0.5385 |
| ast_bpe_v2 | pde | human | complex | 1 | 95.0000 | 0.4847 |
| ast_bpe_v2 | pde | human | simple | 1 | 42.0000 | 0.4941 |
| ast_bpe_v2 | probability | human | simple | 3 | 26.0000 | 0.5909 |
| ast_policy_no_rl | algebra | human | complex | 1 | 112.0000 | 0.7832 |
| ast_policy_no_rl | algebra | human | simple | 12 | 11.9167 | 0.7730 |
| ast_policy_no_rl | analysis | unknown | simple | 2 | 33.0000 | 0.4286 |
| ast_policy_no_rl | combinatorics | possible_mixed | simple | 2 | 23.5000 | 0.8103 |
| ast_policy_no_rl | combinatorics | unknown | simple | 1 | 36.0000 | 0.3396 |
| ast_policy_no_rl | geometry_topology | human | simple | 10 | 13.8000 | 0.6449 |
| ast_policy_no_rl | number_theory | unknown | complex | 1 | 57.0000 | 0.4419 |
| ast_policy_no_rl | optimization | human | simple | 6 | 9.3333 | 0.5385 |
| ast_policy_no_rl | pde | human | complex | 1 | 95.0000 | 0.4847 |
| ast_policy_no_rl | pde | human | simple | 1 | 42.0000 | 0.4941 |
| ast_policy_no_rl | probability | human | simple | 3 | 26.0000 | 0.5909 |
| ast_shortest_path | algebra | human | complex | 1 | 112.0000 | 0.7832 |
| ast_shortest_path | algebra | human | simple | 12 | 11.7500 | 0.7622 |
| ast_shortest_path | analysis | unknown | simple | 2 | 33.0000 | 0.4286 |
| ast_shortest_path | combinatorics | possible_mixed | simple | 2 | 23.5000 | 0.8103 |
| ast_shortest_path | combinatorics | unknown | simple | 1 | 36.0000 | 0.3396 |
| ast_shortest_path | geometry_topology | human | simple | 10 | 13.8000 | 0.6449 |
| ast_shortest_path | number_theory | unknown | complex | 1 | 57.0000 | 0.4419 |
| ast_shortest_path | optimization | human | simple | 6 | 9.3333 | 0.5385 |
| ast_shortest_path | pde | human | complex | 1 | 95.0000 | 0.4847 |
| ast_shortest_path | pde | human | simple | 1 | 42.0000 | 0.4941 |
| ast_shortest_path | probability | human | simple | 3 | 26.0000 | 0.5909 |
| byte_bpe | algebra | human | complex | 1 | 99.0000 | 0.6923 |
| byte_bpe | algebra | human | simple | 12 | 11.3333 | 0.7351 |
| byte_bpe | analysis | unknown | simple | 2 | 28.5000 | 0.3701 |
| byte_bpe | combinatorics | possible_mixed | simple | 2 | 20.5000 | 0.7069 |
| byte_bpe | combinatorics | unknown | simple | 1 | 32.0000 | 0.3019 |
| byte_bpe | geometry_topology | human | simple | 10 | 12.9000 | 0.6028 |
| byte_bpe | number_theory | unknown | complex | 1 | 50.0000 | 0.3876 |
| byte_bpe | optimization | human | simple | 6 | 8.8333 | 0.5096 |
| byte_bpe | pde | human | complex | 1 | 83.0000 | 0.4235 |
| byte_bpe | pde | human | simple | 1 | 38.0000 | 0.4471 |
| byte_bpe | probability | human | simple | 3 | 23.0000 | 0.5227 |
| byte_policy_no_rl | algebra | human | complex | 1 | 99.0000 | 0.6923 |
| byte_policy_no_rl | algebra | human | simple | 12 | 11.3333 | 0.7351 |
| byte_policy_no_rl | analysis | unknown | simple | 2 | 28.5000 | 0.3701 |
| byte_policy_no_rl | combinatorics | possible_mixed | simple | 2 | 20.5000 | 0.7069 |
| byte_policy_no_rl | combinatorics | unknown | simple | 1 | 31.0000 | 0.2925 |
| byte_policy_no_rl | geometry_topology | human | simple | 10 | 12.9000 | 0.6028 |
| byte_policy_no_rl | number_theory | unknown | complex | 1 | 53.0000 | 0.4109 |
| byte_policy_no_rl | optimization | human | simple | 6 | 8.8333 | 0.5096 |
| byte_policy_no_rl | pde | human | complex | 1 | 83.0000 | 0.4235 |
| byte_policy_no_rl | pde | human | simple | 1 | 40.0000 | 0.4706 |
| byte_policy_no_rl | probability | human | simple | 3 | 23.0000 | 0.5227 |
| byte_shortest_path | algebra | human | complex | 1 | 99.0000 | 0.6923 |
| byte_shortest_path | algebra | human | simple | 12 | 11.3333 | 0.7351 |
| byte_shortest_path | analysis | unknown | simple | 2 | 28.5000 | 0.3701 |
| byte_shortest_path | combinatorics | possible_mixed | simple | 2 | 20.5000 | 0.7069 |
| byte_shortest_path | combinatorics | unknown | simple | 1 | 31.0000 | 0.2925 |
| byte_shortest_path | geometry_topology | human | simple | 10 | 12.8000 | 0.5981 |
| byte_shortest_path | number_theory | unknown | complex | 1 | 50.0000 | 0.3876 |
| byte_shortest_path | optimization | human | simple | 6 | 8.8333 | 0.5096 |
| byte_shortest_path | pde | human | complex | 1 | 83.0000 | 0.4235 |
| byte_shortest_path | pde | human | simple | 1 | 38.0000 | 0.4471 |
| byte_shortest_path | probability | human | simple | 3 | 23.0000 | 0.5227 |
| hf_byte_bpe | algebra | human | complex | 1 | 99.0000 | 0.6923 |
| hf_byte_bpe | algebra | human | simple | 12 | 11.5000 | 0.7459 |
| hf_byte_bpe | analysis | unknown | simple | 2 | 28.5000 | 0.3701 |
| hf_byte_bpe | combinatorics | possible_mixed | simple | 2 | 20.5000 | 0.7069 |
| hf_byte_bpe | combinatorics | unknown | simple | 1 | 32.0000 | 0.3019 |
| hf_byte_bpe | geometry_topology | human | simple | 10 | 12.9000 | 0.6028 |
| hf_byte_bpe | number_theory | unknown | complex | 1 | 51.0000 | 0.3953 |
| hf_byte_bpe | optimization | human | simple | 6 | 8.8333 | 0.5096 |
| hf_byte_bpe | pde | human | complex | 1 | 83.0000 | 0.4235 |
| hf_byte_bpe | pde | human | simple | 1 | 38.0000 | 0.4471 |
| hf_byte_bpe | probability | human | simple | 3 | 22.3333 | 0.5076 |
| qwen2.5_official | algebra | human | complex | 1 | 102.0000 | 0.7133 |
| qwen2.5_official | algebra | human | simple | 12 | 10.4167 | 0.6757 |
| qwen2.5_official | analysis | unknown | simple | 2 | 36.0000 | 0.4675 |
| qwen2.5_official | combinatorics | possible_mixed | simple | 2 | 20.5000 | 0.7069 |
| qwen2.5_official | combinatorics | unknown | simple | 1 | 52.0000 | 0.4906 |
| qwen2.5_official | geometry_topology | human | simple | 10 | 12.0000 | 0.5607 |
| qwen2.5_official | number_theory | unknown | complex | 1 | 64.0000 | 0.4961 |
| qwen2.5_official | optimization | human | simple | 6 | 8.0000 | 0.4615 |
| qwen2.5_official | pde | human | complex | 1 | 75.0000 | 0.3827 |
| qwen2.5_official | pde | human | simple | 1 | 33.0000 | 0.3882 |
| qwen2.5_official | probability | human | simple | 3 | 20.6667 | 0.4697 |
| rl_ast_bpe_v1 | algebra | human | complex | 1 | 143.0000 | 1.0000 |
| rl_ast_bpe_v1 | algebra | human | simple | 12 | 14.6667 | 0.9514 |
| rl_ast_bpe_v1 | analysis | unknown | simple | 2 | 69.0000 | 0.8961 |
| rl_ast_bpe_v1 | combinatorics | possible_mixed | simple | 2 | 28.0000 | 0.9655 |
| rl_ast_bpe_v1 | combinatorics | unknown | simple | 1 | 106.0000 | 1.0000 |
| rl_ast_bpe_v1 | geometry_topology | human | simple | 10 | 19.8000 | 0.9252 |
| rl_ast_bpe_v1 | number_theory | unknown | complex | 1 | 129.0000 | 1.0000 |
| rl_ast_bpe_v1 | optimization | human | simple | 6 | 13.8333 | 0.7981 |
| rl_ast_bpe_v1 | pde | human | complex | 1 | 98.0000 | 0.5000 |
| rl_ast_bpe_v1 | pde | human | simple | 1 | 48.0000 | 0.5647 |
| rl_ast_bpe_v1 | probability | human | simple | 3 | 42.3333 | 0.9621 |
| rl_ast_bpe_v2 | algebra | human | complex | 1 | 112.0000 | 0.7832 |
| rl_ast_bpe_v2 | algebra | human | simple | 12 | 11.9167 | 0.7730 |
| rl_ast_bpe_v2 | analysis | unknown | simple | 2 | 33.0000 | 0.4286 |
| rl_ast_bpe_v2 | combinatorics | possible_mixed | simple | 2 | 23.5000 | 0.8103 |
| rl_ast_bpe_v2 | combinatorics | unknown | simple | 1 | 36.0000 | 0.3396 |
| rl_ast_bpe_v2 | geometry_topology | human | simple | 10 | 13.8000 | 0.6449 |
| rl_ast_bpe_v2 | number_theory | unknown | complex | 1 | 57.0000 | 0.4419 |
| rl_ast_bpe_v2 | optimization | human | simple | 6 | 9.3333 | 0.5385 |
| rl_ast_bpe_v2 | pde | human | complex | 1 | 95.0000 | 0.4847 |
| rl_ast_bpe_v2 | pde | human | simple | 1 | 42.0000 | 0.4941 |
| rl_ast_bpe_v2 | probability | human | simple | 3 | 26.0000 | 0.5909 |
| rl_bpe_v1 | algebra | human | complex | 1 | 104.0000 | 0.7273 |
| rl_bpe_v1 | algebra | human | simple | 12 | 11.7500 | 0.7622 |
| rl_bpe_v1 | analysis | unknown | simple | 2 | 28.5000 | 0.3701 |
| rl_bpe_v1 | combinatorics | possible_mixed | simple | 2 | 22.5000 | 0.7759 |
| rl_bpe_v1 | combinatorics | unknown | simple | 1 | 35.0000 | 0.3302 |
| rl_bpe_v1 | geometry_topology | human | simple | 10 | 13.8000 | 0.6449 |
| rl_bpe_v1 | number_theory | unknown | complex | 1 | 52.0000 | 0.4031 |
| rl_bpe_v1 | optimization | human | simple | 6 | 9.0000 | 0.5192 |
| rl_bpe_v1 | pde | human | complex | 1 | 98.0000 | 0.5000 |
| rl_bpe_v1 | pde | human | simple | 1 | 44.0000 | 0.5176 |
| rl_bpe_v1 | probability | human | simple | 3 | 24.3333 | 0.5530 |
| rl_bpe_v2 | algebra | human | complex | 1 | 99.0000 | 0.6923 |
| rl_bpe_v2 | algebra | human | simple | 12 | 11.4167 | 0.7405 |
| rl_bpe_v2 | analysis | unknown | simple | 2 | 28.5000 | 0.3701 |
| rl_bpe_v2 | combinatorics | possible_mixed | simple | 2 | 20.5000 | 0.7069 |
| rl_bpe_v2 | combinatorics | unknown | simple | 1 | 31.0000 | 0.2925 |
| rl_bpe_v2 | geometry_topology | human | simple | 10 | 12.9000 | 0.6028 |
| rl_bpe_v2 | number_theory | unknown | complex | 1 | 51.0000 | 0.3953 |
| rl_bpe_v2 | optimization | human | simple | 6 | 8.8333 | 0.5096 |
| rl_bpe_v2 | pde | human | complex | 1 | 83.0000 | 0.4235 |
| rl_bpe_v2 | pde | human | simple | 1 | 40.0000 | 0.4706 |
| rl_bpe_v2 | probability | human | simple | 3 | 23.0000 | 0.5227 |
| sentencepiece_bpe | algebra | human | complex | 1 | 93.0000 | 0.6503 |
| sentencepiece_bpe | algebra | human | simple | 12 | 11.7500 | 0.7622 |
| sentencepiece_bpe | analysis | unknown | simple | 2 | 34.5000 | 0.4481 |
| sentencepiece_bpe | combinatorics | possible_mixed | simple | 2 | 23.0000 | 0.7931 |
| sentencepiece_bpe | combinatorics | unknown | simple | 1 | 47.0000 | 0.4434 |
| sentencepiece_bpe | geometry_topology | human | simple | 10 | 14.5000 | 0.6776 |
| sentencepiece_bpe | number_theory | unknown | complex | 1 | 60.0000 | 0.4651 |
| sentencepiece_bpe | optimization | human | simple | 6 | 9.8333 | 0.5673 |
| sentencepiece_bpe | pde | human | complex | 1 | 113.0000 | 0.5765 |
| sentencepiece_bpe | pde | human | simple | 1 | 49.0000 | 0.5765 |
| sentencepiece_bpe | probability | human | simple | 3 | 24.6667 | 0.5606 |
| sentencepiece_unigram | algebra | human | complex | 1 | 95.0000 | 0.6643 |
| sentencepiece_unigram | algebra | human | simple | 12 | 12.4167 | 0.8054 |
| sentencepiece_unigram | analysis | unknown | simple | 2 | 40.0000 | 0.5195 |
| sentencepiece_unigram | combinatorics | possible_mixed | simple | 2 | 23.5000 | 0.8103 |
| sentencepiece_unigram | combinatorics | unknown | simple | 1 | 50.0000 | 0.4717 |
| sentencepiece_unigram | geometry_topology | human | simple | 10 | 14.1000 | 0.6589 |
| sentencepiece_unigram | number_theory | unknown | complex | 1 | 68.0000 | 0.5271 |
| sentencepiece_unigram | optimization | human | simple | 6 | 10.0000 | 0.5769 |
| sentencepiece_unigram | pde | human | complex | 1 | 88.0000 | 0.4490 |
| sentencepiece_unigram | pde | human | simple | 1 | 36.0000 | 0.4235 |
| sentencepiece_unigram | probability | human | simple | 3 | 25.6667 | 0.5833 |
| tiktoken_cl100k_base | algebra | human | complex | 1 | 89.0000 | 0.6224 |
| tiktoken_cl100k_base | algebra | human | simple | 12 | 10.4167 | 0.6757 |
| tiktoken_cl100k_base | analysis | unknown | simple | 2 | 36.0000 | 0.4675 |
| tiktoken_cl100k_base | combinatorics | possible_mixed | simple | 2 | 20.0000 | 0.6897 |
| tiktoken_cl100k_base | combinatorics | unknown | simple | 1 | 52.0000 | 0.4906 |
| tiktoken_cl100k_base | geometry_topology | human | simple | 10 | 11.5000 | 0.5374 |
| tiktoken_cl100k_base | number_theory | unknown | complex | 1 | 64.0000 | 0.4961 |
| tiktoken_cl100k_base | optimization | human | simple | 6 | 8.0000 | 0.4615 |
| tiktoken_cl100k_base | pde | human | complex | 1 | 75.0000 | 0.3827 |
| tiktoken_cl100k_base | pde | human | simple | 1 | 33.0000 | 0.3882 |
| tiktoken_cl100k_base | probability | human | simple | 3 | 20.6667 | 0.4697 |
| tiktoken_gpt2 | algebra | human | complex | 1 | 90.0000 | 0.6294 |
| tiktoken_gpt2 | algebra | human | simple | 12 | 11.0833 | 0.7189 |
| tiktoken_gpt2 | analysis | unknown | simple | 2 | 40.5000 | 0.5260 |
| tiktoken_gpt2 | combinatorics | possible_mixed | simple | 2 | 21.5000 | 0.7414 |
| tiktoken_gpt2 | combinatorics | unknown | simple | 1 | 51.0000 | 0.4811 |
| tiktoken_gpt2 | geometry_topology | human | simple | 10 | 13.9000 | 0.6495 |
| tiktoken_gpt2 | number_theory | unknown | complex | 1 | 66.0000 | 0.5116 |
| tiktoken_gpt2 | optimization | human | simple | 6 | 10.8333 | 0.6250 |
| tiktoken_gpt2 | pde | human | complex | 1 | 75.0000 | 0.3827 |
| tiktoken_gpt2 | pde | human | simple | 1 | 35.0000 | 0.4118 |
| tiktoken_gpt2 | probability | human | simple | 3 | 25.3333 | 0.5758 |
| tiktoken_o200k_base | algebra | human | complex | 1 | 89.0000 | 0.6224 |
| tiktoken_o200k_base | algebra | human | simple | 12 | 10.3333 | 0.6703 |
| tiktoken_o200k_base | analysis | unknown | simple | 2 | 35.5000 | 0.4610 |
| tiktoken_o200k_base | combinatorics | possible_mixed | simple | 2 | 19.0000 | 0.6552 |
| tiktoken_o200k_base | combinatorics | unknown | simple | 1 | 52.0000 | 0.4906 |
| tiktoken_o200k_base | geometry_topology | human | simple | 10 | 11.8000 | 0.5514 |
| tiktoken_o200k_base | number_theory | unknown | complex | 1 | 63.0000 | 0.4884 |
| tiktoken_o200k_base | optimization | human | simple | 6 | 7.6667 | 0.4423 |
| tiktoken_o200k_base | pde | human | complex | 1 | 75.0000 | 0.3827 |
| tiktoken_o200k_base | pde | human | simple | 1 | 35.0000 | 0.4118 |
| tiktoken_o200k_base | probability | human | simple | 3 | 20.3333 | 0.4621 |
| utf8_bytes | algebra | human | complex | 1 | 143.0000 | 1.0000 |
| utf8_bytes | algebra | human | simple | 12 | 15.4167 | 1.0000 |
| utf8_bytes | analysis | unknown | simple | 2 | 77.0000 | 1.0000 |
| utf8_bytes | combinatorics | possible_mixed | simple | 2 | 29.0000 | 1.0000 |
| utf8_bytes | combinatorics | unknown | simple | 1 | 106.0000 | 1.0000 |
| utf8_bytes | geometry_topology | human | simple | 10 | 21.4000 | 1.0000 |
| utf8_bytes | number_theory | unknown | complex | 1 | 129.0000 | 1.0000 |
| utf8_bytes | optimization | human | simple | 6 | 17.3333 | 1.0000 |
| utf8_bytes | pde | human | complex | 1 | 196.0000 | 1.0000 |
| utf8_bytes | pde | human | simple | 1 | 85.0000 | 1.0000 |
| utf8_bytes | probability | human | simple | 3 | 44.0000 | 1.0000 |

## 组配对置信区间

| 比较 | token/公式差值 | 文档/家族bootstrap95%区间 | 组数 |
|---|---:|---|---:|
| rl_bpe_v2-vs-byte_bpe/17 | 0.1000 | -0.0556, 0.3602 | 15 |
| rl_ast_bpe_v2-vs-ast_bpe_v2/17 | 0.0500 | 0.0000, 0.1200 | 15 |
| rl_bpe_v2-vs-byte_bpe/29 | 0.1000 | -0.0626, 0.2609 | 15 |
| rl_ast_bpe_v2-vs-ast_bpe_v2/29 | 0.0500 | 0.0000, 0.1200 | 15 |
| rl_bpe_v2-vs-byte_bpe/43 | 0.0250 | -0.0769, 0.1668 | 15 |
| rl_ast_bpe_v2-vs-ast_bpe_v2/43 | 0.0500 | 0.0000, 0.1200 | 15 |

## 指标的用途与限制

指标细节、方向、相关论文与官方实现见docs/intrinsic_metrics_v2.md。Rényi效率、熵、empirical unigram bits/byte都只是tokenizer-only描述；经验unigram编码成本不是语言模型BPB。
固定预训练词表单列，不用不同词表和外部语料得出的压缩率作为公平训练算法胜负。算法使用相同数据与vocab上限；实际训练算力有差别，秒数和策略训练样本量透明记录。
测试公式全部参与，每条有结果；没有因结构解析失败移除困难公式。结构解析可以缓存用于指标分析，但每个方法每轮速度测试前清缓存，比较的是完整解析+编码成本。
2020年前论文按用户批准的历史来源规则标human，证据强度另存，不声称逐作者无自动化辅助认证。Fungrim来源不确定性保留，机械符号转LaTeX与LLM生成区分记录。
本版未训练下游模型，不报告推理、PDE识别或语言模型性能改善。
