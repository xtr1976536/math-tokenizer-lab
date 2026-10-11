# 第三版优化：固定语料、无新基准

词表预算512，共同测试3959条。数学语料和split哈希冻结；仅重测现有自研方法以取得同进程/同输入的速度比较。没有新训练成熟基准。
配置由dev选择：atom128、atom64-relative12-cross1.5。原生代码首次编译不计入稳态encode；每轮清结构缓存，实际结构解析计时。

| 方法 | token/公式 | 命令完整率 | 数字完整率 | lexeme保留率 | 跨树比例 | Rényi α=2 | 编码MiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ast_bpe_v2 | 20.1018 | 0.7559 | 0.9355 | 0.9598 | 0.0000 | 0.6456 | 0.5391 |
| ast_bpe_v3 | 18.5090 | 0.9650 | 0.9355 | 0.9895 | 0.0000 | 0.6444 | 1.1790 |
| atomic_byte_shortest_control | 17.0482 | 0.8669 | 0.9355 | 0.9740 | 0.1464 | 0.7668 | 1.1831 |
| rl_bpe_v2 | 18.6763 | 0.5310 | 0.9355 | 0.9260 | 0.1429 | 0.7601 | 0.0969 |
| rl_bpe_v2_native_exact | 18.6763 | 0.5310 | 0.9355 | 0.9260 | 0.1429 | 0.7601 | 1.1676 |
| rl_bpe_v3 | 17.2916 | 0.8683 | 0.9355 | 0.9758 | 0.1317 | 0.7556 | 1.1718 |
| rl_ast_bpe_v2 | 19.9667 | 0.7622 | 0.9355 | 0.9608 | 0.0000 | 0.6443 | 0.0822 |
| rl_ast_bpe_v3 | 18.5289 | 0.9650 | 0.9355 | 0.9895 | 0.0000 | 0.6407 | 1.1764 |

## 配对文档置信区间

| 比较 | token/公式差值 | 95%组bootstrap |
|---|---:|---|
| rl_bpe_v3-vs-rl_bpe_v2/17 | -1.3872 | -1.6823, -1.1528 |
| rl_ast_bpe_v3-vs-rl_ast_bpe_v2/17 | -1.4377 | -1.8079, -1.1004 |
| rl_bpe_v3-vs-rl_bpe_v2/29 | -1.4976 | -1.8357, -1.2490 |
| rl_ast_bpe_v3-vs-rl_ast_bpe_v2/29 | -1.4377 | -1.8079, -1.1004 |
| rl_bpe_v3-vs-rl_bpe_v2/43 | -1.2693 | -1.5570, -1.0315 |
| rl_ast_bpe_v3-vs-rl_ast_bpe_v2/43 | -1.4377 | -1.8079, -1.1004 |
| ast_bpe_v3-vs-v2 | -1.5928 | -2.0996, -1.1705 |

## 解释

AST-v3仍先构建结构，训练词表时预留训练集常见LaTeX命令原子，之后做受源跨度约束的BPE。推断改为受结构约束的最短路径，因此不是纯ranked BPE；词表训练与推断改动的收益不能都归因于AST本身。
RL-v3复用原始REINFORCE路径策略，命令原子种子+原生推断+验证选择的soft tree penalty；RL已训练、概率和参数保存。相同新词表no-RL shortest控制用于拆分词表/搜索与策略学习收益。
exact native模式与v2输出一致，用于单独验证速度优化。原生kernel用于所有v3模型，Python语法结构解析保持一致；端到端速度仍受parser影响。
更高命令完整性并不必然使Rényi熵、词表利用率等所有指标同向改善，完整JSON保留指标取舍，不把未改善指标隐藏。
测试集是此前已查看的开发test，配置仍只按dev选择；本轮不得称最终未见测试结论。模型质量提升与代码实现速度提升分别解释。

## 分层

| 方法 | 复杂度 | 领域 | 公式数 | token/公式 |
|---|---|---|---:|---:|
| ast_bpe_v2 | complex | algebra | 17 | 74.8824 |
| ast_bpe_v2 | complex | analysis | 258 | 53.6434 |
| ast_bpe_v2 | complex | combinatorics | 73 | 57.8904 |
| ast_bpe_v2 | complex | geometry_topology | 16 | 56.6250 |
| ast_bpe_v2 | complex | linear_algebra | 8 | 60.8750 |
| ast_bpe_v2 | complex | mathematical_physics | 79 | 76.1392 |
| ast_bpe_v2 | complex | number_theory | 5 | 74.6000 |
| ast_bpe_v2 | complex | ode | 20 | 59.0000 |
| ast_bpe_v2 | complex | optimization | 26 | 60.5000 |
| ast_bpe_v2 | complex | pde | 29 | 51.7241 |
| ast_bpe_v2 | complex | probability | 45 | 59.8000 |
| ast_bpe_v2 | simple | algebra | 232 | 12.6810 |
| ast_bpe_v2 | simple | analysis | 1240 | 13.1629 |
| ast_bpe_v2 | simple | combinatorics | 297 | 14.2222 |
| ast_bpe_v2 | simple | geometry_topology | 367 | 13.1253 |
| ast_bpe_v2 | simple | linear_algebra | 5 | 11.2000 |
| ast_bpe_v2 | simple | mathematical_physics | 594 | 12.6717 |
| ast_bpe_v2 | simple | number_theory | 58 | 14.3276 |
| ast_bpe_v2 | simple | ode | 209 | 15.3206 |
| ast_bpe_v2 | simple | optimization | 108 | 15.3796 |
| ast_bpe_v2 | simple | pde | 60 | 19.6333 |
| ast_bpe_v2 | simple | probability | 213 | 12.9531 |
| ast_bpe_v3 | complex | algebra | 17 | 72.1176 |
| ast_bpe_v3 | complex | analysis | 258 | 49.8721 |
| ast_bpe_v3 | complex | combinatorics | 73 | 53.3973 |
| ast_bpe_v3 | complex | geometry_topology | 16 | 53.5000 |
| ast_bpe_v3 | complex | linear_algebra | 8 | 57.0000 |
| ast_bpe_v3 | complex | mathematical_physics | 79 | 67.2152 |
| ast_bpe_v3 | complex | number_theory | 5 | 62.0000 |
| ast_bpe_v3 | complex | ode | 20 | 50.0000 |
| ast_bpe_v3 | complex | optimization | 26 | 56.3077 |
| ast_bpe_v3 | complex | pde | 29 | 50.1724 |
| ast_bpe_v3 | complex | probability | 45 | 52.1111 |
| ast_bpe_v3 | simple | algebra | 232 | 11.3922 |
| ast_bpe_v3 | simple | analysis | 1240 | 12.4444 |
| ast_bpe_v3 | simple | combinatorics | 297 | 13.1717 |
| ast_bpe_v3 | simple | geometry_topology | 367 | 12.4278 |
| ast_bpe_v3 | simple | linear_algebra | 5 | 11.0000 |
| ast_bpe_v3 | simple | mathematical_physics | 594 | 11.3333 |
| ast_bpe_v3 | simple | number_theory | 58 | 13.2241 |
| ast_bpe_v3 | simple | ode | 209 | 13.9426 |
| ast_bpe_v3 | simple | optimization | 108 | 14.2500 |
| ast_bpe_v3 | simple | pde | 60 | 18.3333 |
| ast_bpe_v3 | simple | probability | 213 | 11.4366 |
| atomic_byte_shortest_control | complex | algebra | 17 | 65.2941 |
| atomic_byte_shortest_control | complex | analysis | 258 | 45.1512 |
| atomic_byte_shortest_control | complex | combinatorics | 73 | 48.2740 |
| atomic_byte_shortest_control | complex | geometry_topology | 16 | 49.5000 |
| atomic_byte_shortest_control | complex | linear_algebra | 8 | 54.1250 |
| atomic_byte_shortest_control | complex | mathematical_physics | 79 | 62.6329 |
| atomic_byte_shortest_control | complex | number_theory | 5 | 54.8000 |
| atomic_byte_shortest_control | complex | ode | 20 | 45.2500 |
| atomic_byte_shortest_control | complex | optimization | 26 | 49.6154 |
| atomic_byte_shortest_control | complex | pde | 29 | 36.6207 |
| atomic_byte_shortest_control | complex | probability | 45 | 50.2444 |
| atomic_byte_shortest_control | simple | algebra | 232 | 11.5000 |
| atomic_byte_shortest_control | simple | analysis | 1240 | 11.5637 |
| atomic_byte_shortest_control | simple | combinatorics | 297 | 12.4377 |
| atomic_byte_shortest_control | simple | geometry_topology | 367 | 10.9237 |
| atomic_byte_shortest_control | simple | linear_algebra | 5 | 9.6000 |
| atomic_byte_shortest_control | simple | mathematical_physics | 594 | 10.8906 |
| atomic_byte_shortest_control | simple | number_theory | 58 | 12.1207 |
| atomic_byte_shortest_control | simple | ode | 209 | 13.1244 |
| atomic_byte_shortest_control | simple | optimization | 108 | 12.3426 |
| atomic_byte_shortest_control | simple | pde | 60 | 14.9000 |
| atomic_byte_shortest_control | simple | probability | 213 | 11.0141 |
| rl_ast_bpe_v2 | complex | algebra | 17 | 74.8235 |
| rl_ast_bpe_v2 | complex | analysis | 258 | 53.5891 |
| rl_ast_bpe_v2 | complex | combinatorics | 73 | 57.8493 |
| rl_ast_bpe_v2 | complex | geometry_topology | 16 | 56.6875 |
| rl_ast_bpe_v2 | complex | linear_algebra | 8 | 60.6250 |
| rl_ast_bpe_v2 | complex | mathematical_physics | 79 | 72.6203 |
| rl_ast_bpe_v2 | complex | number_theory | 5 | 74.6000 |
| rl_ast_bpe_v2 | complex | ode | 20 | 58.6000 |
| rl_ast_bpe_v2 | complex | optimization | 26 | 60.4231 |
| rl_ast_bpe_v2 | complex | pde | 29 | 51.7241 |
| rl_ast_bpe_v2 | complex | probability | 45 | 59.8222 |
| rl_ast_bpe_v2 | simple | algebra | 232 | 12.6767 |
| rl_ast_bpe_v2 | simple | analysis | 1240 | 13.1468 |
| rl_ast_bpe_v2 | simple | combinatorics | 297 | 14.2256 |
| rl_ast_bpe_v2 | simple | geometry_topology | 367 | 13.1090 |
| rl_ast_bpe_v2 | simple | linear_algebra | 5 | 11.2000 |
| rl_ast_bpe_v2 | simple | mathematical_physics | 594 | 12.3855 |
| rl_ast_bpe_v2 | simple | number_theory | 58 | 14.3448 |
| rl_ast_bpe_v2 | simple | ode | 209 | 15.2679 |
| rl_ast_bpe_v2 | simple | optimization | 108 | 15.2130 |
| rl_ast_bpe_v2 | simple | pde | 60 | 19.6333 |
| rl_ast_bpe_v2 | simple | probability | 213 | 12.9296 |
| rl_ast_bpe_v3 | complex | algebra | 17 | 72.1176 |
| rl_ast_bpe_v3 | complex | analysis | 258 | 49.9147 |
| rl_ast_bpe_v3 | complex | combinatorics | 73 | 53.3973 |
| rl_ast_bpe_v3 | complex | geometry_topology | 16 | 53.5000 |
| rl_ast_bpe_v3 | complex | linear_algebra | 8 | 57.0000 |
| rl_ast_bpe_v3 | complex | mathematical_physics | 79 | 67.8987 |
| rl_ast_bpe_v3 | complex | number_theory | 5 | 62.0000 |
| rl_ast_bpe_v3 | complex | ode | 20 | 50.0000 |
| rl_ast_bpe_v3 | complex | optimization | 26 | 56.3077 |
| rl_ast_bpe_v3 | complex | pde | 29 | 50.1724 |
| rl_ast_bpe_v3 | complex | probability | 45 | 52.1111 |
| rl_ast_bpe_v3 | simple | algebra | 232 | 11.3922 |
| rl_ast_bpe_v3 | simple | analysis | 1240 | 12.4492 |
| rl_ast_bpe_v3 | simple | combinatorics | 297 | 13.1785 |
| rl_ast_bpe_v3 | simple | geometry_topology | 367 | 12.4278 |
| rl_ast_bpe_v3 | simple | linear_algebra | 5 | 11.0000 |
| rl_ast_bpe_v3 | simple | mathematical_physics | 594 | 11.3434 |
| rl_ast_bpe_v3 | simple | number_theory | 58 | 13.2241 |
| rl_ast_bpe_v3 | simple | ode | 209 | 13.9426 |
| rl_ast_bpe_v3 | simple | optimization | 108 | 14.2500 |
| rl_ast_bpe_v3 | simple | pde | 60 | 18.3333 |
| rl_ast_bpe_v3 | simple | probability | 213 | 11.4366 |
| rl_bpe_v2 | complex | algebra | 17 | 70.5882 |
| rl_bpe_v2 | complex | analysis | 258 | 49.5026 |
| rl_bpe_v2 | complex | combinatorics | 73 | 54.7260 |
| rl_bpe_v2 | complex | geometry_topology | 16 | 53.8750 |
| rl_bpe_v2 | complex | linear_algebra | 8 | 59.0000 |
| rl_bpe_v2 | complex | mathematical_physics | 79 | 69.2911 |
| rl_bpe_v2 | complex | number_theory | 5 | 63.6000 |
| rl_bpe_v2 | complex | ode | 20 | 48.1500 |
| rl_bpe_v2 | complex | optimization | 26 | 56.3462 |
| rl_bpe_v2 | complex | pde | 29 | 44.0345 |
| rl_bpe_v2 | complex | probability | 45 | 55.5556 |
| rl_bpe_v2 | simple | algebra | 232 | 12.4138 |
| rl_bpe_v2 | simple | analysis | 1240 | 12.5040 |
| rl_bpe_v2 | simple | combinatorics | 297 | 13.1987 |
| rl_bpe_v2 | simple | geometry_topology | 367 | 11.7620 |
| rl_bpe_v2 | simple | linear_algebra | 5 | 10.0000 |
| rl_bpe_v2 | simple | mathematical_physics | 594 | 12.0118 |
| rl_bpe_v2 | simple | number_theory | 58 | 13.0345 |
| rl_bpe_v2 | simple | ode | 209 | 13.8804 |
| rl_bpe_v2 | simple | optimization | 108 | 14.2130 |
| rl_bpe_v2 | simple | pde | 60 | 17.3000 |
| rl_bpe_v2 | simple | probability | 213 | 12.2300 |
| rl_bpe_v2_native_exact | complex | algebra | 17 | 70.5882 |
| rl_bpe_v2_native_exact | complex | analysis | 258 | 49.5026 |
| rl_bpe_v2_native_exact | complex | combinatorics | 73 | 54.7260 |
| rl_bpe_v2_native_exact | complex | geometry_topology | 16 | 53.8750 |
| rl_bpe_v2_native_exact | complex | linear_algebra | 8 | 59.0000 |
| rl_bpe_v2_native_exact | complex | mathematical_physics | 79 | 69.2911 |
| rl_bpe_v2_native_exact | complex | number_theory | 5 | 63.6000 |
| rl_bpe_v2_native_exact | complex | ode | 20 | 48.1500 |
| rl_bpe_v2_native_exact | complex | optimization | 26 | 56.3462 |
| rl_bpe_v2_native_exact | complex | pde | 29 | 44.0345 |
| rl_bpe_v2_native_exact | complex | probability | 45 | 55.5556 |
| rl_bpe_v2_native_exact | simple | algebra | 232 | 12.4138 |
| rl_bpe_v2_native_exact | simple | analysis | 1240 | 12.5040 |
| rl_bpe_v2_native_exact | simple | combinatorics | 297 | 13.1987 |
| rl_bpe_v2_native_exact | simple | geometry_topology | 367 | 11.7620 |
| rl_bpe_v2_native_exact | simple | linear_algebra | 5 | 10.0000 |
| rl_bpe_v2_native_exact | simple | mathematical_physics | 594 | 12.0118 |
| rl_bpe_v2_native_exact | simple | number_theory | 58 | 13.0345 |
| rl_bpe_v2_native_exact | simple | ode | 209 | 13.8804 |
| rl_bpe_v2_native_exact | simple | optimization | 108 | 14.2130 |
| rl_bpe_v2_native_exact | simple | pde | 60 | 17.3000 |
| rl_bpe_v2_native_exact | simple | probability | 213 | 12.2300 |
| rl_bpe_v3 | complex | algebra | 17 | 66.5490 |
| rl_bpe_v3 | complex | analysis | 258 | 46.2080 |
| rl_bpe_v3 | complex | combinatorics | 73 | 49.3516 |
| rl_bpe_v3 | complex | geometry_topology | 16 | 49.6042 |
| rl_bpe_v3 | complex | linear_algebra | 8 | 54.3750 |
| rl_bpe_v3 | complex | mathematical_physics | 79 | 64.2489 |
| rl_bpe_v3 | complex | number_theory | 5 | 57.0667 |
| rl_bpe_v3 | complex | ode | 20 | 46.3667 |
| rl_bpe_v3 | complex | optimization | 26 | 51.1282 |
| rl_bpe_v3 | complex | pde | 29 | 36.8736 |
| rl_bpe_v3 | complex | probability | 45 | 50.4963 |
| rl_bpe_v3 | simple | algebra | 232 | 11.5718 |
| rl_bpe_v3 | simple | analysis | 1240 | 11.7169 |
| rl_bpe_v3 | simple | combinatorics | 297 | 12.5185 |
| rl_bpe_v3 | simple | geometry_topology | 367 | 10.9964 |
| rl_bpe_v3 | simple | linear_algebra | 5 | 9.6000 |
| rl_bpe_v3 | simple | mathematical_physics | 594 | 10.9461 |
| rl_bpe_v3 | simple | number_theory | 58 | 12.4080 |
| rl_bpe_v3 | simple | ode | 209 | 13.2855 |
| rl_bpe_v3 | simple | optimization | 108 | 12.5093 |
| rl_bpe_v3 | simple | pde | 60 | 15.0111 |
| rl_bpe_v3 | simple | probability | 213 | 11.0250 |
