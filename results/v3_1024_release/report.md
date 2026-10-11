# 第三版优化：固定语料、无新基准

词表预算1024，共同测试3959条。数学语料和split哈希冻结；仅重测现有自研方法以取得同进程/同输入的速度比较。没有新训练成熟基准。
配置由dev选择：atom128、atom128-rloo24-cross1.5。原生代码首次编译不计入稳态encode；每轮清结构缓存，实际结构解析计时。

| 方法 | token/公式 | 命令完整率 | 数字完整率 | lexeme保留率 | 跨树比例 | Rényi α=2 | 编码MiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ast_bpe_v2 | 15.8032 | 0.9205 | 0.9483 | 0.9841 | 0.0000 | 0.6578 | 0.4678 |
| ast_bpe_v3 | 14.9752 | 0.9650 | 0.9490 | 0.9906 | 0.0000 | 0.6753 | 1.1707 |
| atomic_byte_shortest_control | 13.9172 | 0.9615 | 0.9436 | 0.9877 | 0.1901 | 0.7684 | 1.1573 |
| rl_bpe_v2 | 14.9190 | 0.8144 | 0.9355 | 0.9656 | 0.1864 | 0.7597 | 0.0919 |
| rl_bpe_v2_native_exact | 14.9190 | 0.8144 | 0.9355 | 0.9656 | 0.1864 | 0.7597 | 1.1459 |
| rl_bpe_v3 | 13.9361 | 0.9650 | 0.9431 | 0.9901 | 0.1818 | 0.7639 | 1.1479 |
| rl_ast_bpe_v2 | 15.6443 | 0.9269 | 0.9483 | 0.9851 | 0.0000 | 0.6577 | 0.0775 |
| rl_ast_bpe_v3 | 15.0062 | 0.9650 | 0.9490 | 0.9906 | 0.0000 | 0.6725 | 1.1527 |

## 配对文档置信区间

| 比较 | token/公式差值 | 95%组bootstrap |
|---|---:|---|
| rl_bpe_v3-vs-rl_bpe_v2/17 | -0.9755 | -1.3127, -0.7229 |
| rl_ast_bpe_v3-vs-rl_ast_bpe_v2/17 | -0.6401 | -0.9354, -0.4477 |
| rl_bpe_v3-vs-rl_bpe_v2/29 | -0.9917 | -1.3264, -0.7440 |
| rl_ast_bpe_v3-vs-rl_ast_bpe_v2/29 | -0.6345 | -0.9297, -0.4430 |
| rl_bpe_v3-vs-rl_bpe_v2/43 | -0.9816 | -1.3172, -0.7308 |
| rl_ast_bpe_v3-vs-rl_ast_bpe_v2/43 | -0.6396 | -0.9372, -0.4465 |
| ast_bpe_v3-vs-v2 | -0.8280 | -1.3394, -0.5022 |

## 解释

AST-v3仍先构建结构，训练词表时预留训练集常见LaTeX命令原子，之后做受源跨度约束的BPE。推断改为受结构约束的最短路径，因此不是纯ranked BPE；词表训练与推断改动的收益不能都归因于AST本身。
RL-v3复用原始REINFORCE路径策略，命令原子种子+原生推断+验证选择的soft tree penalty；RL已训练、概率和参数保存。相同新词表no-RL shortest控制用于拆分词表/搜索与策略学习收益。
exact native模式与v2输出一致，用于单独验证速度优化。原生kernel用于所有v3模型，Python语法结构解析保持一致；端到端速度仍受parser影响。
更高命令完整性并不必然使Rényi熵、词表利用率等所有指标同向改善，完整JSON保留指标取舍，不把未改善指标隐藏。
测试集是此前已查看的开发test，配置仍只按dev选择；本轮不得称最终未见测试结论。模型质量提升与代码实现速度提升分别解释。

## 分层

| 方法 | 复杂度 | 领域 | 公式数 | token/公式 |
|---|---|---|---:|---:|
| ast_bpe_v2 | complex | algebra | 17 | 61.2941 |
| ast_bpe_v2 | complex | analysis | 258 | 41.7907 |
| ast_bpe_v2 | complex | combinatorics | 73 | 44.1644 |
| ast_bpe_v2 | complex | geometry_topology | 16 | 45.7500 |
| ast_bpe_v2 | complex | linear_algebra | 8 | 46.3750 |
| ast_bpe_v2 | complex | mathematical_physics | 79 | 60.7722 |
| ast_bpe_v2 | complex | number_theory | 5 | 53.0000 |
| ast_bpe_v2 | complex | ode | 20 | 43.5000 |
| ast_bpe_v2 | complex | optimization | 26 | 45.2692 |
| ast_bpe_v2 | complex | pde | 29 | 35.0345 |
| ast_bpe_v2 | complex | probability | 45 | 43.1333 |
| ast_bpe_v2 | simple | algebra | 232 | 9.7457 |
| ast_bpe_v2 | simple | analysis | 1240 | 10.8460 |
| ast_bpe_v2 | simple | combinatorics | 297 | 11.1684 |
| ast_bpe_v2 | simple | geometry_topology | 367 | 10.2916 |
| ast_bpe_v2 | simple | linear_algebra | 5 | 9.8000 |
| ast_bpe_v2 | simple | mathematical_physics | 594 | 10.2104 |
| ast_bpe_v2 | simple | number_theory | 58 | 11.6034 |
| ast_bpe_v2 | simple | ode | 209 | 12.1053 |
| ast_bpe_v2 | simple | optimization | 108 | 12.0185 |
| ast_bpe_v2 | simple | pde | 60 | 13.4167 |
| ast_bpe_v2 | simple | probability | 213 | 9.9531 |
| ast_bpe_v3 | complex | algebra | 17 | 59.0000 |
| ast_bpe_v3 | complex | analysis | 258 | 39.1395 |
| ast_bpe_v3 | complex | combinatorics | 73 | 40.1918 |
| ast_bpe_v3 | complex | geometry_topology | 16 | 45.3125 |
| ast_bpe_v3 | complex | linear_algebra | 8 | 42.2500 |
| ast_bpe_v3 | complex | mathematical_physics | 79 | 53.6329 |
| ast_bpe_v3 | complex | number_theory | 5 | 51.0000 |
| ast_bpe_v3 | complex | ode | 20 | 42.3000 |
| ast_bpe_v3 | complex | optimization | 26 | 44.1538 |
| ast_bpe_v3 | complex | pde | 29 | 33.0690 |
| ast_bpe_v3 | complex | probability | 45 | 40.5778 |
| ast_bpe_v3 | simple | algebra | 232 | 9.3060 |
| ast_bpe_v3 | simple | analysis | 1240 | 10.5363 |
| ast_bpe_v3 | simple | combinatorics | 297 | 10.2963 |
| ast_bpe_v3 | simple | geometry_topology | 367 | 10.3270 |
| ast_bpe_v3 | simple | linear_algebra | 5 | 9.8000 |
| ast_bpe_v3 | simple | mathematical_physics | 594 | 9.4983 |
| ast_bpe_v3 | simple | number_theory | 58 | 11.0517 |
| ast_bpe_v3 | simple | ode | 209 | 11.7656 |
| ast_bpe_v3 | simple | optimization | 108 | 11.4630 |
| ast_bpe_v3 | simple | pde | 60 | 12.6000 |
| ast_bpe_v3 | simple | probability | 213 | 9.6761 |
| atomic_byte_shortest_control | complex | algebra | 17 | 52.5294 |
| atomic_byte_shortest_control | complex | analysis | 258 | 35.9225 |
| atomic_byte_shortest_control | complex | combinatorics | 73 | 38.5342 |
| atomic_byte_shortest_control | complex | geometry_topology | 16 | 41.9375 |
| atomic_byte_shortest_control | complex | linear_algebra | 8 | 40.0000 |
| atomic_byte_shortest_control | complex | mathematical_physics | 79 | 49.9367 |
| atomic_byte_shortest_control | complex | number_theory | 5 | 48.2000 |
| atomic_byte_shortest_control | complex | ode | 20 | 38.8000 |
| atomic_byte_shortest_control | complex | optimization | 26 | 41.6923 |
| atomic_byte_shortest_control | complex | pde | 29 | 27.0000 |
| atomic_byte_shortest_control | complex | probability | 45 | 39.0444 |
| atomic_byte_shortest_control | simple | algebra | 232 | 8.9784 |
| atomic_byte_shortest_control | simple | analysis | 1240 | 9.7274 |
| atomic_byte_shortest_control | simple | combinatorics | 297 | 9.9226 |
| atomic_byte_shortest_control | simple | geometry_topology | 367 | 9.5777 |
| atomic_byte_shortest_control | simple | linear_algebra | 5 | 9.0000 |
| atomic_byte_shortest_control | simple | mathematical_physics | 594 | 8.8064 |
| atomic_byte_shortest_control | simple | number_theory | 58 | 10.0172 |
| atomic_byte_shortest_control | simple | ode | 209 | 11.1005 |
| atomic_byte_shortest_control | simple | optimization | 108 | 10.7222 |
| atomic_byte_shortest_control | simple | pde | 60 | 10.8833 |
| atomic_byte_shortest_control | simple | probability | 213 | 9.1643 |
| rl_ast_bpe_v2 | complex | algebra | 17 | 61.1176 |
| rl_ast_bpe_v2 | complex | analysis | 258 | 41.7855 |
| rl_ast_bpe_v2 | complex | combinatorics | 73 | 44.1233 |
| rl_ast_bpe_v2 | complex | geometry_topology | 16 | 45.6875 |
| rl_ast_bpe_v2 | complex | linear_algebra | 8 | 46.3750 |
| rl_ast_bpe_v2 | complex | mathematical_physics | 79 | 55.8903 |
| rl_ast_bpe_v2 | complex | number_theory | 5 | 53.0000 |
| rl_ast_bpe_v2 | complex | ode | 20 | 43.5000 |
| rl_ast_bpe_v2 | complex | optimization | 26 | 45.3846 |
| rl_ast_bpe_v2 | complex | pde | 29 | 34.9655 |
| rl_ast_bpe_v2 | complex | probability | 45 | 43.2222 |
| rl_ast_bpe_v2 | simple | algebra | 232 | 9.6897 |
| rl_ast_bpe_v2 | simple | analysis | 1240 | 10.8296 |
| rl_ast_bpe_v2 | simple | combinatorics | 297 | 11.1515 |
| rl_ast_bpe_v2 | simple | geometry_topology | 367 | 10.3243 |
| rl_ast_bpe_v2 | simple | linear_algebra | 5 | 9.8000 |
| rl_ast_bpe_v2 | simple | mathematical_physics | 594 | 9.8670 |
| rl_ast_bpe_v2 | simple | number_theory | 58 | 11.6034 |
| rl_ast_bpe_v2 | simple | ode | 209 | 12.1340 |
| rl_ast_bpe_v2 | simple | optimization | 108 | 11.8086 |
| rl_ast_bpe_v2 | simple | pde | 60 | 13.4333 |
| rl_ast_bpe_v2 | simple | probability | 213 | 9.9797 |
| rl_ast_bpe_v3 | complex | algebra | 17 | 59.0000 |
| rl_ast_bpe_v3 | complex | analysis | 258 | 39.3915 |
| rl_ast_bpe_v3 | complex | combinatorics | 73 | 40.2055 |
| rl_ast_bpe_v3 | complex | geometry_topology | 16 | 45.3125 |
| rl_ast_bpe_v3 | complex | linear_algebra | 8 | 42.5000 |
| rl_ast_bpe_v3 | complex | mathematical_physics | 79 | 53.6582 |
| rl_ast_bpe_v3 | complex | number_theory | 5 | 51.0000 |
| rl_ast_bpe_v3 | complex | ode | 20 | 42.3000 |
| rl_ast_bpe_v3 | complex | optimization | 26 | 44.2692 |
| rl_ast_bpe_v3 | complex | pde | 29 | 33.4713 |
| rl_ast_bpe_v3 | complex | probability | 45 | 40.6889 |
| rl_ast_bpe_v3 | simple | algebra | 232 | 9.3060 |
| rl_ast_bpe_v3 | simple | analysis | 1240 | 10.5452 |
| rl_ast_bpe_v3 | simple | combinatorics | 297 | 10.3098 |
| rl_ast_bpe_v3 | simple | geometry_topology | 367 | 10.3297 |
| rl_ast_bpe_v3 | simple | linear_algebra | 5 | 9.8000 |
| rl_ast_bpe_v3 | simple | mathematical_physics | 594 | 9.5067 |
| rl_ast_bpe_v3 | simple | number_theory | 58 | 11.0690 |
| rl_ast_bpe_v3 | simple | ode | 209 | 11.7703 |
| rl_ast_bpe_v3 | simple | optimization | 108 | 11.4630 |
| rl_ast_bpe_v3 | simple | pde | 60 | 12.6500 |
| rl_ast_bpe_v3 | simple | probability | 213 | 9.7089 |
| rl_bpe_v2 | complex | algebra | 17 | 53.0588 |
| rl_bpe_v2 | complex | analysis | 258 | 39.5181 |
| rl_bpe_v2 | complex | combinatorics | 73 | 42.5799 |
| rl_bpe_v2 | complex | geometry_topology | 16 | 44.8542 |
| rl_bpe_v2 | complex | linear_algebra | 8 | 44.5000 |
| rl_bpe_v2 | complex | mathematical_physics | 79 | 54.5274 |
| rl_bpe_v2 | complex | number_theory | 5 | 48.6000 |
| rl_bpe_v2 | complex | ode | 20 | 39.3500 |
| rl_bpe_v2 | complex | optimization | 26 | 44.1154 |
| rl_bpe_v2 | complex | pde | 29 | 30.7931 |
| rl_bpe_v2 | complex | probability | 45 | 44.2370 |
| rl_bpe_v2 | simple | algebra | 232 | 9.3276 |
| rl_bpe_v2 | simple | analysis | 1240 | 10.3780 |
| rl_bpe_v2 | simple | combinatorics | 297 | 10.4837 |
| rl_bpe_v2 | simple | geometry_topology | 367 | 9.7711 |
| rl_bpe_v2 | simple | linear_algebra | 5 | 9.0000 |
| rl_bpe_v2 | simple | mathematical_physics | 594 | 9.4697 |
| rl_bpe_v2 | simple | number_theory | 58 | 10.3621 |
| rl_bpe_v2 | simple | ode | 209 | 11.4689 |
| rl_bpe_v2 | simple | optimization | 108 | 10.9846 |
| rl_bpe_v2 | simple | pde | 60 | 12.0667 |
| rl_bpe_v2 | simple | probability | 213 | 9.8858 |
| rl_bpe_v2_native_exact | complex | algebra | 17 | 53.0588 |
| rl_bpe_v2_native_exact | complex | analysis | 258 | 39.5181 |
| rl_bpe_v2_native_exact | complex | combinatorics | 73 | 42.5799 |
| rl_bpe_v2_native_exact | complex | geometry_topology | 16 | 44.8542 |
| rl_bpe_v2_native_exact | complex | linear_algebra | 8 | 44.5000 |
| rl_bpe_v2_native_exact | complex | mathematical_physics | 79 | 54.5274 |
| rl_bpe_v2_native_exact | complex | number_theory | 5 | 48.6000 |
| rl_bpe_v2_native_exact | complex | ode | 20 | 39.3500 |
| rl_bpe_v2_native_exact | complex | optimization | 26 | 44.1154 |
| rl_bpe_v2_native_exact | complex | pde | 29 | 30.7931 |
| rl_bpe_v2_native_exact | complex | probability | 45 | 44.2370 |
| rl_bpe_v2_native_exact | simple | algebra | 232 | 9.3276 |
| rl_bpe_v2_native_exact | simple | analysis | 1240 | 10.3780 |
| rl_bpe_v2_native_exact | simple | combinatorics | 297 | 10.4837 |
| rl_bpe_v2_native_exact | simple | geometry_topology | 367 | 9.7711 |
| rl_bpe_v2_native_exact | simple | linear_algebra | 5 | 9.0000 |
| rl_bpe_v2_native_exact | simple | mathematical_physics | 594 | 9.4697 |
| rl_bpe_v2_native_exact | simple | number_theory | 58 | 10.3621 |
| rl_bpe_v2_native_exact | simple | ode | 209 | 11.4689 |
| rl_bpe_v2_native_exact | simple | optimization | 108 | 10.9846 |
| rl_bpe_v2_native_exact | simple | pde | 60 | 12.0667 |
| rl_bpe_v2_native_exact | simple | probability | 213 | 9.8858 |
| rl_bpe_v3 | complex | algebra | 17 | 52.5294 |
| rl_bpe_v3 | complex | analysis | 258 | 35.9574 |
| rl_bpe_v3 | complex | combinatorics | 73 | 38.6164 |
| rl_bpe_v3 | complex | geometry_topology | 16 | 41.9375 |
| rl_bpe_v3 | complex | linear_algebra | 8 | 40.0000 |
| rl_bpe_v3 | complex | mathematical_physics | 79 | 49.9620 |
| rl_bpe_v3 | complex | number_theory | 5 | 48.2000 |
| rl_bpe_v3 | complex | ode | 20 | 38.8000 |
| rl_bpe_v3 | complex | optimization | 26 | 41.6923 |
| rl_bpe_v3 | complex | pde | 29 | 27.1379 |
| rl_bpe_v3 | complex | probability | 45 | 39.1111 |
| rl_bpe_v3 | simple | algebra | 232 | 8.9784 |
| rl_bpe_v3 | simple | analysis | 1240 | 9.7605 |
| rl_bpe_v3 | simple | combinatorics | 297 | 9.9461 |
| rl_bpe_v3 | simple | geometry_topology | 367 | 9.5804 |
| rl_bpe_v3 | simple | linear_algebra | 5 | 9.0000 |
| rl_bpe_v3 | simple | mathematical_physics | 594 | 8.8064 |
| rl_bpe_v3 | simple | number_theory | 58 | 10.0172 |
| rl_bpe_v3 | simple | ode | 209 | 11.1053 |
| rl_bpe_v3 | simple | optimization | 108 | 10.7315 |
| rl_bpe_v3 | simple | pde | 60 | 10.8833 |
| rl_bpe_v3 | simple | probability | 213 | 9.1643 |
