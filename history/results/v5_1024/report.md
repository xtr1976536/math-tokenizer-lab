# 新一轮：RL词表、多目标奖励与泛化

v5语料53858条，新增25511条、119篇固定许可arXiv源码。旧v2保持不变，新test hash先冻结。词表预算1024，正式三种子17/29/43。
RL学习merge/table，推断使用官方Rust BPE。没有hard AST或手工命令清单，reward与候选统计仍需设计。复合奖励包含code cost/边界/rare/空白变体，不单纯token数；所有reward来自train内部paper holdout。
配置dev选择：{'broad': {'config': 'composite_robust', 'dev_scores': {'composite_balanced': 0.3474486352897902, 'composite_structure': 0.33420190884085343, 'composite_code': 0.35706969204416533, 'composite_robust': 0.32607881001265376}}, 'specialized': {'config': 'composite_robust', 'dev_scores': {'composite_balanced': 0.25727864145560503, 'composite_structure': 0.2397272108756415, 'composite_code': 0.2608498352970432, 'composite_robust': 0.23195385251373937}}}。两场景都冻结selection后才读fresh test，测试成绩未用来选择或修改配置。

| 训练场景 | 测试集 | 方法 | token/公式 | tokens/byte | 命令完整率 | tree crossing | Rényi2 | 编码MiB/s | 无损率 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| broad | fresh_paper_test | frequency | 22.1778 | 0.3705 | 0.7826 | 0.2572 | 0.7284 | 9.0044 | 1.0000 |
| broad | fresh_paper_test | hf_bpe_fulltrain | 22.3751 | 0.3738 | 0.7626 | 0.2592 | 0.7277 | 8.6436 | 1.0000 |
| broad | fresh_paper_test | hf_bpe_matched_fit | 22.1778 | 0.3705 | 0.7826 | 0.2572 | 0.7284 | 8.5901 | 1.0000 |
| broad | fresh_paper_test | random | 23.5623 | 0.3936 | 0.4827 | 0.3009 | 0.7335 | 8.8256 | 1.0000 |
| broad | fresh_paper_test | reward_greedy | 21.5458 | 0.3599 | 0.8989 | 0.2568 | 0.7288 | 8.8563 | 1.0000 |
| broad | fresh_paper_test | rl_composite | 21.7221 | 0.3629 | 0.8775 | 0.2622 | 0.7223 | 8.6835 | 1.0000 |
| broad | fresh_paper_test | rl_length_only | 22.7858 | 0.3806 | 0.6432 | 0.2826 | 0.7349 | 8.7285 | 1.0000 |
| broad | fresh_paper_test | sp_bpe | 27.9790 | 0.4674 | 0.0000 | 0.2523 | 0.6191 | 11.2085 | 1.0000 |
| broad | fresh_paper_test | sp_unigram | 29.0197 | 0.4848 | 0.0000 | 0.1914 | 0.5392 | 22.0904 | 1.0000 |
| broad | synthetic_stress_test | frequency | 103.5440 | 0.4455 | 0.8556 | 0.3585 | 0.4885 | 8.9858 | 1.0000 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | 102.9221 | 0.4429 | 0.8423 | 0.3620 | 0.4899 | 8.6743 | 1.0000 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | 103.5440 | 0.4455 | 0.8556 | 0.3585 | 0.4885 | 8.8387 | 1.0000 |
| broad | synthetic_stress_test | random | 105.7100 | 0.4549 | 0.6259 | 0.3741 | 0.4950 | 8.7360 | 1.0000 |
| broad | synthetic_stress_test | reward_greedy | 102.1443 | 0.4395 | 0.9190 | 0.3609 | 0.4910 | 7.5927 | 1.0000 |
| broad | synthetic_stress_test | rl_composite | 102.9149 | 0.4428 | 0.9077 | 0.3598 | 0.4902 | 8.6275 | 1.0000 |
| broad | synthetic_stress_test | rl_length_only | 103.4719 | 0.4452 | 0.7791 | 0.3689 | 0.4948 | 8.7007 | 1.0000 |
| broad | synthetic_stress_test | sp_bpe | 118.1515 | 0.5084 | 0.0000 | 0.2940 | 0.5076 | 11.5111 | 1.0000 |
| broad | synthetic_stress_test | sp_unigram | 121.3261 | 0.5220 | 0.0000 | 0.2647 | 0.4731 | 36.5534 | 1.0000 |
| prior_data_reference | extreme_domain_test | legacy_v3_rl | 13.9334 | 0.4234 | 0.9695 | 0.1527 | 0.7249 | 1.0318 | 1.0000 |
| prior_data_reference | extreme_domain_test | legacy_v4_masked_rl | 15.0966 | 0.4587 | 1.0000 | 0.0000 | 0.6345 | 0.9690 | 1.0000 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | 21.1273 | 0.3529 | 0.9846 | 0.2265 | 0.7016 | 1.4008 | 1.0000 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | 23.2439 | 0.3883 | 0.9996 | 0.0000 | 0.6347 | 1.3147 | 1.0000 |
| reference | extreme_domain_test | tiktoken_cl100k_base | 18.0086 | 0.5472 | 0.0000 | 0.2226 | 0.3591 | 9.0759 | 1.0000 |
| reference | extreme_domain_test | tiktoken_o200k_base | 18.0648 | 0.5489 | 0.0000 | 0.2235 | 0.3429 | 12.8572 | 1.0000 |
| reference | fresh_paper_test | tiktoken_cl100k_base | 31.1159 | 0.5198 | 0.0000 | 0.2202 | 0.3389 | 10.0638 | 1.0000 |
| reference | fresh_paper_test | tiktoken_o200k_base | 30.7191 | 0.5132 | 0.0000 | 0.2247 | 0.3256 | 13.2376 | 1.0000 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | 121.4358 | 0.5225 | 0.0000 | 0.3885 | 0.2813 | 13.0912 | 1.0000 |
| reference | synthetic_stress_test | tiktoken_o200k_base | 112.8874 | 0.4857 | 0.0000 | 0.4193 | 0.2681 | 19.3702 | 1.0000 |
| specialized | extreme_domain_test | frequency | 16.0043 | 0.4863 | 0.4768 | 0.1899 | 0.7123 | 7.1701 | 1.0000 |
| specialized | extreme_domain_test | hf_bpe_fulltrain | 15.7139 | 0.4775 | 0.5863 | 0.1620 | 0.7098 | 7.3874 | 1.0000 |
| specialized | extreme_domain_test | hf_bpe_matched_fit | 16.0043 | 0.4863 | 0.4768 | 0.1899 | 0.7123 | 7.3393 | 1.0000 |
| specialized | extreme_domain_test | random | 15.5685 | 0.4730 | 0.7344 | 0.1493 | 0.7029 | 6.7616 | 1.0000 |
| specialized | extreme_domain_test | reward_greedy | 15.5012 | 0.4710 | 0.7469 | 0.1538 | 0.7010 | 7.3608 | 1.0000 |
| specialized | extreme_domain_test | rl_composite | 15.5892 | 0.4737 | 0.7186 | 0.1503 | 0.7039 | 7.1121 | 1.0000 |
| specialized | extreme_domain_test | rl_length_only | 15.7274 | 0.4779 | 0.5224 | 0.1705 | 0.7104 | 7.0912 | 1.0000 |
| specialized | extreme_domain_test | sp_bpe | 17.0795 | 0.5189 | 0.0000 | 0.2132 | 0.6677 | 10.7923 | 1.0000 |
| specialized | extreme_domain_test | sp_unigram | 16.7604 | 0.5092 | 0.0000 | 0.1847 | 0.6305 | 17.9709 | 1.0000 |
| specialized | synthetic_stress_test | frequency | 109.1039 | 0.4695 | 0.7938 | 0.3092 | 0.4792 | 8.6366 | 1.0000 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | 108.0361 | 0.4649 | 0.8334 | 0.3105 | 0.4779 | 8.8503 | 1.0000 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | 109.1039 | 0.4695 | 0.7938 | 0.3092 | 0.4792 | 8.6965 | 1.0000 |
| specialized | synthetic_stress_test | random | 112.9856 | 0.4862 | 0.8621 | 0.2954 | 0.4487 | 8.8984 | 1.0000 |
| specialized | synthetic_stress_test | reward_greedy | 112.7316 | 0.4851 | 0.8816 | 0.2931 | 0.4662 | 9.0393 | 1.0000 |
| specialized | synthetic_stress_test | rl_composite | 109.2973 | 0.4703 | 0.8853 | 0.2986 | 0.4726 | 8.8974 | 1.0000 |
| specialized | synthetic_stress_test | rl_length_only | 109.1241 | 0.4695 | 0.7897 | 0.3068 | 0.4744 | 8.9398 | 1.0000 |
| specialized | synthetic_stress_test | sp_bpe | 118.5123 | 0.5099 | 0.0000 | 0.2905 | 0.5086 | 11.6424 | 1.0000 |
| specialized | synthetic_stress_test | sp_unigram | 119.8456 | 0.5157 | 0.0000 | 0.2799 | 0.4834 | 34.2851 | 1.0000 |

## 泛化任务的含义

fresh_paper_test：新增从未参与旧实验的论文，按paper隔离，排除旧release和新train/dev相同词法模板。broad训练包含多个领域，因此这是新文档泛化，不是未见领域。
extreme_domain_test：specialized只训练PDE/ODE/analysis，排除algebra/geometry/number theory/combinatorics/logic及cross-list源类别；再测新论文这些目标领域。这是本轮真正的领域迁移；与broad不同训练集/数据量，不能把差异全归因于域。
synthetic_stress_test：heldout公式去空白/字母重命名/长式组合，以及2048项、深度256、rare Unicode/未知macro/巨大数字/损坏TeX。人工生成变体明确标synthetic，不能当自然科学公式或真值标签。

## RL贡献与奖励消融

frequency/random/reward_greedy使用相同内部fit数据；HF matched-fit控制查明数据差异，HF full-train查实际成熟方法。reward greedy可能强于RL，不隐瞒；RL length-only是同路线奖励消融。
奖励函数与policy是contextual合并，仍没有实现长时合并规划。统计features可以包含内部holdout各reward分量，不需要泛化到在线输入，因为policy只在离线学词表使用；最终官方BPE合并表是部署产物。
当前全期望bandit gradient是确定性学习器：如果seed17/29/43给相同merge表，属于重复确定性运行，不是三个独立随机发现。
这里“减少人介入”是删除手工命令词表/AST动作mask/逐式标注，不是删除设计者对reward、budget、feature和候选生成的选择。无法保证所有指标都优秀。

## 速度解释

RLBPE和HF共用官方Rust实现、相同无regex预切分，速度差距主要来自学到的merge table/token分布。SentencePiece也是成熟原生实现，但算法不同；同数据、同vocab、同per-formula接口5次中位。
不运行parser/policy意味着推断不再受Python结构前端拖慢。新算法是离线RL词表生成，不冒称同一v4路径策略的纯代码加速。

## 统计区间

| 比较 | token/公式差值 | 95%paper bootstrap |
|---|---:|---|
| broad/fresh_paper_test/rl-vs-frequency/17 | -0.4558 | -0.8306, 0.0546 |
| broad/fresh_paper_test/rl-vs-rl_length_only/17 | -1.0638 | -1.4645, -0.5535 |
| broad/fresh_paper_test/rl-vs-reward_greedy/17 | 0.1762 | -0.1713, 0.4478 |
| broad/fresh_paper_test/rl-vs-hf_bpe_fulltrain/17 | -0.6530 | -1.0212, -0.2572 |
| broad/fresh_paper_test/rl-vs-hf_bpe_matched_fit/17 | -0.4558 | -0.8306, 0.0546 |
| broad/fresh_paper_test/rl-vs-frequency/29 | -0.4558 | -0.8306, 0.0546 |
| broad/fresh_paper_test/rl-vs-rl_length_only/29 | -1.0638 | -1.4645, -0.5535 |
| broad/fresh_paper_test/rl-vs-reward_greedy/29 | 0.1762 | -0.1713, 0.4478 |
| broad/fresh_paper_test/rl-vs-hf_bpe_fulltrain/29 | -0.6530 | -1.0212, -0.2572 |
| broad/fresh_paper_test/rl-vs-hf_bpe_matched_fit/29 | -0.4558 | -0.8306, 0.0546 |
| broad/fresh_paper_test/rl-vs-frequency/43 | -0.4558 | -0.8306, 0.0546 |
| broad/fresh_paper_test/rl-vs-rl_length_only/43 | -1.0638 | -1.4645, -0.5535 |
| broad/fresh_paper_test/rl-vs-reward_greedy/43 | 0.1762 | -0.1713, 0.4478 |
| broad/fresh_paper_test/rl-vs-hf_bpe_fulltrain/43 | -0.6530 | -1.0212, -0.2572 |
| broad/fresh_paper_test/rl-vs-hf_bpe_matched_fit/43 | -0.4558 | -0.8306, 0.0546 |
| broad/synthetic_stress_test/rl-vs-frequency/17 | -0.6291 | -1.0216, -0.0430 |
| broad/synthetic_stress_test/rl-vs-rl_length_only/17 | -0.5570 | -1.3501, 0.6711 |
| broad/synthetic_stress_test/rl-vs-reward_greedy/17 | 0.7706 | -0.3701, 2.1796 |
| broad/synthetic_stress_test/rl-vs-hf_bpe_fulltrain/17 | -0.0072 | -1.0404, 1.7729 |
| broad/synthetic_stress_test/rl-vs-hf_bpe_matched_fit/17 | -0.6291 | -1.0216, -0.0430 |
| broad/synthetic_stress_test/rl-vs-frequency/29 | -0.6291 | -1.0216, -0.0430 |
| broad/synthetic_stress_test/rl-vs-rl_length_only/29 | -0.5570 | -1.3501, 0.6711 |
| broad/synthetic_stress_test/rl-vs-reward_greedy/29 | 0.7706 | -0.3701, 2.1796 |
| broad/synthetic_stress_test/rl-vs-hf_bpe_fulltrain/29 | -0.0072 | -1.0404, 1.7729 |
| broad/synthetic_stress_test/rl-vs-hf_bpe_matched_fit/29 | -0.6291 | -1.0216, -0.0430 |
| broad/synthetic_stress_test/rl-vs-frequency/43 | -0.6291 | -1.0216, -0.0430 |
| broad/synthetic_stress_test/rl-vs-rl_length_only/43 | -0.5570 | -1.3501, 0.6711 |
| broad/synthetic_stress_test/rl-vs-reward_greedy/43 | 0.7706 | -0.3701, 2.1796 |
| broad/synthetic_stress_test/rl-vs-hf_bpe_fulltrain/43 | -0.0072 | -1.0404, 1.7729 |
| broad/synthetic_stress_test/rl-vs-hf_bpe_matched_fit/43 | -0.6291 | -1.0216, -0.0430 |
| specialized/extreme_domain_test/rl-vs-frequency/17 | -0.4150 | -0.8917, -0.0059 |
| specialized/extreme_domain_test/rl-vs-rl_length_only/17 | -0.1381 | -0.4251, 0.1197 |
| specialized/extreme_domain_test/rl-vs-reward_greedy/17 | 0.0880 | -0.0507, 0.2433 |
| specialized/extreme_domain_test/rl-vs-hf_bpe_fulltrain/17 | -0.1247 | -0.3343, 0.0999 |
| specialized/extreme_domain_test/rl-vs-hf_bpe_matched_fit/17 | -0.4150 | -0.8917, -0.0059 |
| specialized/extreme_domain_test/rl-vs-frequency/29 | -0.4150 | -0.8917, -0.0059 |
| specialized/extreme_domain_test/rl-vs-rl_length_only/29 | -0.1381 | -0.4251, 0.1197 |
| specialized/extreme_domain_test/rl-vs-reward_greedy/29 | 0.0880 | -0.0507, 0.2433 |
| specialized/extreme_domain_test/rl-vs-hf_bpe_fulltrain/29 | -0.1247 | -0.3343, 0.0999 |
| specialized/extreme_domain_test/rl-vs-hf_bpe_matched_fit/29 | -0.4150 | -0.8917, -0.0059 |
| specialized/extreme_domain_test/rl-vs-frequency/43 | -0.4150 | -0.8917, -0.0059 |
| specialized/extreme_domain_test/rl-vs-rl_length_only/43 | -0.1381 | -0.4251, 0.1197 |
| specialized/extreme_domain_test/rl-vs-reward_greedy/43 | 0.0880 | -0.0507, 0.2433 |
| specialized/extreme_domain_test/rl-vs-hf_bpe_fulltrain/43 | -0.1247 | -0.3343, 0.0999 |
| specialized/extreme_domain_test/rl-vs-hf_bpe_matched_fit/43 | -0.4150 | -0.8917, -0.0059 |
| specialized/synthetic_stress_test/rl-vs-frequency/17 | 0.1934 | -0.9541, 1.9887 |
| specialized/synthetic_stress_test/rl-vs-rl_length_only/17 | 0.1732 | -0.5756, 1.1272 |
| specialized/synthetic_stress_test/rl-vs-reward_greedy/17 | -3.4343 | -9.9085, 0.1499 |
| specialized/synthetic_stress_test/rl-vs-hf_bpe_fulltrain/17 | 1.2612 | -0.4434, 3.8025 |
| specialized/synthetic_stress_test/rl-vs-hf_bpe_matched_fit/17 | 0.1934 | -0.9541, 1.9887 |
| specialized/synthetic_stress_test/rl-vs-frequency/29 | 0.1934 | -0.9541, 1.9887 |
| specialized/synthetic_stress_test/rl-vs-rl_length_only/29 | 0.1732 | -0.5756, 1.1272 |
| specialized/synthetic_stress_test/rl-vs-reward_greedy/29 | -3.4343 | -9.9085, 0.1499 |
| specialized/synthetic_stress_test/rl-vs-hf_bpe_fulltrain/29 | 1.2612 | -0.4434, 3.8025 |
| specialized/synthetic_stress_test/rl-vs-hf_bpe_matched_fit/29 | 0.1934 | -0.9541, 1.9887 |
| specialized/synthetic_stress_test/rl-vs-frequency/43 | 0.1934 | -0.9541, 1.9887 |
| specialized/synthetic_stress_test/rl-vs-rl_length_only/43 | 0.1732 | -0.5756, 1.1272 |
| specialized/synthetic_stress_test/rl-vs-reward_greedy/43 | -3.4343 | -9.9085, 0.1499 |
| specialized/synthetic_stress_test/rl-vs-hf_bpe_fulltrain/43 | 1.2612 | -0.4434, 3.8025 |
| specialized/synthetic_stress_test/rl-vs-hf_bpe_matched_fit/43 | 0.1934 | -0.9541, 1.9887 |

## 按领域与stress类型

| 场景 | 集合 | 方法 | 子群 | 公式数 | token/byte |
|---|---|---|---|---:|---:|
| broad | fresh_paper_test | frequency | algebra | 327 | 0.4560 |
| broad | fresh_paper_test | frequency | analysis | 1568 | 0.3608 |
| broad | fresh_paper_test | frequency | combinatorics | 472 | 0.4875 |
| broad | fresh_paper_test | frequency | geometry_topology | 500 | 0.4051 |
| broad | fresh_paper_test | frequency | linear_algebra | 13 | 0.4599 |
| broad | fresh_paper_test | frequency | logic | 46 | 0.4132 |
| broad | fresh_paper_test | frequency | mathematical_physics | 248 | 0.4406 |
| broad | fresh_paper_test | frequency | number_theory | 291 | 0.4979 |
| broad | fresh_paper_test | frequency | ode | 98 | 0.4331 |
| broad | fresh_paper_test | frequency | optimization | 235 | 0.3805 |
| broad | fresh_paper_test | frequency | pde | 31 | 0.3888 |
| broad | fresh_paper_test | frequency | probability | 123 | 0.4872 |
| broad | fresh_paper_test | frequency | statistics | 423 | 0.2807 |
| broad | fresh_paper_test | hf_bpe_fulltrain | algebra | 327 | 0.4567 |
| broad | fresh_paper_test | hf_bpe_fulltrain | analysis | 1568 | 0.3629 |
| broad | fresh_paper_test | hf_bpe_fulltrain | combinatorics | 472 | 0.4905 |
| broad | fresh_paper_test | hf_bpe_fulltrain | geometry_topology | 500 | 0.4033 |
| broad | fresh_paper_test | hf_bpe_fulltrain | linear_algebra | 13 | 0.4705 |
| broad | fresh_paper_test | hf_bpe_fulltrain | logic | 46 | 0.3938 |
| broad | fresh_paper_test | hf_bpe_fulltrain | mathematical_physics | 248 | 0.4419 |
| broad | fresh_paper_test | hf_bpe_fulltrain | number_theory | 291 | 0.4928 |
| broad | fresh_paper_test | hf_bpe_fulltrain | ode | 98 | 0.4316 |
| broad | fresh_paper_test | hf_bpe_fulltrain | optimization | 235 | 0.3874 |
| broad | fresh_paper_test | hf_bpe_fulltrain | pde | 31 | 0.3757 |
| broad | fresh_paper_test | hf_bpe_fulltrain | probability | 123 | 0.5018 |
| broad | fresh_paper_test | hf_bpe_fulltrain | statistics | 423 | 0.2910 |
| broad | fresh_paper_test | hf_bpe_matched_fit | algebra | 327 | 0.4560 |
| broad | fresh_paper_test | hf_bpe_matched_fit | analysis | 1568 | 0.3608 |
| broad | fresh_paper_test | hf_bpe_matched_fit | combinatorics | 472 | 0.4875 |
| broad | fresh_paper_test | hf_bpe_matched_fit | geometry_topology | 500 | 0.4051 |
| broad | fresh_paper_test | hf_bpe_matched_fit | linear_algebra | 13 | 0.4599 |
| broad | fresh_paper_test | hf_bpe_matched_fit | logic | 46 | 0.4132 |
| broad | fresh_paper_test | hf_bpe_matched_fit | mathematical_physics | 248 | 0.4406 |
| broad | fresh_paper_test | hf_bpe_matched_fit | number_theory | 291 | 0.4979 |
| broad | fresh_paper_test | hf_bpe_matched_fit | ode | 98 | 0.4331 |
| broad | fresh_paper_test | hf_bpe_matched_fit | optimization | 235 | 0.3805 |
| broad | fresh_paper_test | hf_bpe_matched_fit | pde | 31 | 0.3888 |
| broad | fresh_paper_test | hf_bpe_matched_fit | probability | 123 | 0.4872 |
| broad | fresh_paper_test | hf_bpe_matched_fit | statistics | 423 | 0.2807 |
| broad | fresh_paper_test | random | algebra | 327 | 0.4695 |
| broad | fresh_paper_test | random | analysis | 1568 | 0.3795 |
| broad | fresh_paper_test | random | combinatorics | 472 | 0.5160 |
| broad | fresh_paper_test | random | geometry_topology | 500 | 0.4263 |
| broad | fresh_paper_test | random | linear_algebra | 13 | 0.4735 |
| broad | fresh_paper_test | random | logic | 46 | 0.4396 |
| broad | fresh_paper_test | random | mathematical_physics | 248 | 0.4645 |
| broad | fresh_paper_test | random | number_theory | 291 | 0.5173 |
| broad | fresh_paper_test | random | ode | 98 | 0.4438 |
| broad | fresh_paper_test | random | optimization | 235 | 0.3920 |
| broad | fresh_paper_test | random | pde | 31 | 0.3871 |
| broad | fresh_paper_test | random | probability | 123 | 0.5177 |
| broad | fresh_paper_test | random | statistics | 423 | 0.3197 |
| broad | fresh_paper_test | reward_greedy | algebra | 327 | 0.4432 |
| broad | fresh_paper_test | reward_greedy | analysis | 1568 | 0.3424 |
| broad | fresh_paper_test | reward_greedy | combinatorics | 472 | 0.4816 |
| broad | fresh_paper_test | reward_greedy | geometry_topology | 500 | 0.4010 |
| broad | fresh_paper_test | reward_greedy | linear_algebra | 13 | 0.4478 |
| broad | fresh_paper_test | reward_greedy | logic | 46 | 0.4238 |
| broad | fresh_paper_test | reward_greedy | mathematical_physics | 248 | 0.4311 |
| broad | fresh_paper_test | reward_greedy | number_theory | 291 | 0.4945 |
| broad | fresh_paper_test | reward_greedy | ode | 98 | 0.4260 |
| broad | fresh_paper_test | reward_greedy | optimization | 235 | 0.3703 |
| broad | fresh_paper_test | reward_greedy | pde | 31 | 0.3830 |
| broad | fresh_paper_test | reward_greedy | probability | 123 | 0.4905 |
| broad | fresh_paper_test | reward_greedy | statistics | 423 | 0.2804 |
| broad | fresh_paper_test | rl_composite | algebra | 327 | 0.4397 |
| broad | fresh_paper_test | rl_composite | analysis | 1568 | 0.3458 |
| broad | fresh_paper_test | rl_composite | combinatorics | 472 | 0.4786 |
| broad | fresh_paper_test | rl_composite | geometry_topology | 500 | 0.3923 |
| broad | fresh_paper_test | rl_composite | linear_algebra | 13 | 0.4523 |
| broad | fresh_paper_test | rl_composite | logic | 46 | 0.4185 |
| broad | fresh_paper_test | rl_composite | mathematical_physics | 248 | 0.4352 |
| broad | fresh_paper_test | rl_composite | number_theory | 291 | 0.4999 |
| broad | fresh_paper_test | rl_composite | ode | 98 | 0.4237 |
| broad | fresh_paper_test | rl_composite | optimization | 235 | 0.3748 |
| broad | fresh_paper_test | rl_composite | pde | 31 | 0.3688 |
| broad | fresh_paper_test | rl_composite | probability | 123 | 0.4926 |
| broad | fresh_paper_test | rl_composite | statistics | 423 | 0.2904 |
| broad | fresh_paper_test | rl_length_only | algebra | 327 | 0.4547 |
| broad | fresh_paper_test | rl_length_only | analysis | 1568 | 0.3637 |
| broad | fresh_paper_test | rl_length_only | combinatorics | 472 | 0.5097 |
| broad | fresh_paper_test | rl_length_only | geometry_topology | 500 | 0.4100 |
| broad | fresh_paper_test | rl_length_only | linear_algebra | 13 | 0.4644 |
| broad | fresh_paper_test | rl_length_only | logic | 46 | 0.4264 |
| broad | fresh_paper_test | rl_length_only | mathematical_physics | 248 | 0.4448 |
| broad | fresh_paper_test | rl_length_only | number_theory | 291 | 0.4998 |
| broad | fresh_paper_test | rl_length_only | ode | 98 | 0.4321 |
| broad | fresh_paper_test | rl_length_only | optimization | 235 | 0.3868 |
| broad | fresh_paper_test | rl_length_only | pde | 31 | 0.3835 |
| broad | fresh_paper_test | rl_length_only | probability | 123 | 0.4973 |
| broad | fresh_paper_test | rl_length_only | statistics | 423 | 0.3128 |
| broad | fresh_paper_test | sp_bpe | algebra | 327 | 0.5097 |
| broad | fresh_paper_test | sp_bpe | analysis | 1568 | 0.4537 |
| broad | fresh_paper_test | sp_bpe | combinatorics | 472 | 0.5215 |
| broad | fresh_paper_test | sp_bpe | geometry_topology | 500 | 0.4825 |
| broad | fresh_paper_test | sp_bpe | linear_algebra | 13 | 0.5779 |
| broad | fresh_paper_test | sp_bpe | logic | 46 | 0.4987 |
| broad | fresh_paper_test | sp_bpe | mathematical_physics | 248 | 0.4597 |
| broad | fresh_paper_test | sp_bpe | number_theory | 291 | 0.5584 |
| broad | fresh_paper_test | sp_bpe | ode | 98 | 0.5392 |
| broad | fresh_paper_test | sp_bpe | optimization | 235 | 0.4857 |
| broad | fresh_paper_test | sp_bpe | pde | 31 | 0.4298 |
| broad | fresh_paper_test | sp_bpe | probability | 123 | 0.5350 |
| broad | fresh_paper_test | sp_bpe | statistics | 423 | 0.4422 |
| broad | fresh_paper_test | sp_unigram | algebra | 327 | 0.5179 |
| broad | fresh_paper_test | sp_unigram | analysis | 1568 | 0.4842 |
| broad | fresh_paper_test | sp_unigram | combinatorics | 472 | 0.4856 |
| broad | fresh_paper_test | sp_unigram | geometry_topology | 500 | 0.4871 |
| broad | fresh_paper_test | sp_unigram | linear_algebra | 13 | 0.5613 |
| broad | fresh_paper_test | sp_unigram | logic | 46 | 0.5357 |
| broad | fresh_paper_test | sp_unigram | mathematical_physics | 248 | 0.4503 |
| broad | fresh_paper_test | sp_unigram | number_theory | 291 | 0.5727 |
| broad | fresh_paper_test | sp_unigram | ode | 98 | 0.5549 |
| broad | fresh_paper_test | sp_unigram | optimization | 235 | 0.5060 |
| broad | fresh_paper_test | sp_unigram | pde | 31 | 0.4344 |
| broad | fresh_paper_test | sp_unigram | probability | 123 | 0.5191 |
| broad | fresh_paper_test | sp_unigram | statistics | 423 | 0.4595 |
| broad | synthetic_stress_test | frequency | deep_nesting_16 | 1 | 0.3043 |
| broad | synthetic_stress_test | frequency | deep_nesting_256 | 1 | 0.3003 |
| broad | synthetic_stress_test | frequency | deep_nesting_64 | 1 | 0.3011 |
| broad | synthetic_stress_test | frequency | huge_numbers | 1 | 0.9999 |
| broad | synthetic_stress_test | frequency | long_composition_2 | 256 | 0.3833 |
| broad | synthetic_stress_test | frequency | long_formula_terms_128 | 1 | 0.4115 |
| broad | synthetic_stress_test | frequency | long_formula_terms_2048 | 1 | 0.4447 |
| broad | synthetic_stress_test | frequency | long_formula_terms_32 | 1 | 0.3729 |
| broad | synthetic_stress_test | frequency | long_formula_terms_512 | 1 | 0.4243 |
| broad | synthetic_stress_test | frequency | malformed_tex | 1 | 0.3333 |
| broad | synthetic_stress_test | frequency | rare_unicode | 1 | 0.9231 |
| broad | synthetic_stress_test | frequency | symbol_renaming | 239 | 0.4035 |
| broad | synthetic_stress_test | frequency | unknown_macro_unicode | 1 | 0.7708 |
| broad | synthetic_stress_test | frequency | whitespace_removed | 187 | 0.3595 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | deep_nesting_16 | 1 | 0.3043 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | deep_nesting_256 | 1 | 0.3003 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | deep_nesting_64 | 1 | 0.3011 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | huge_numbers | 1 | 0.9999 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | long_composition_2 | 256 | 0.3864 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | long_formula_terms_128 | 1 | 0.4067 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | long_formula_terms_2048 | 1 | 0.4417 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | long_formula_terms_32 | 1 | 0.3729 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | long_formula_terms_512 | 1 | 0.4115 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | malformed_tex | 1 | 0.3333 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | rare_unicode | 1 | 0.9231 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | symbol_renaming | 239 | 0.3994 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | unknown_macro_unicode | 1 | 0.8125 |
| broad | synthetic_stress_test | hf_bpe_fulltrain | whitespace_removed | 187 | 0.3589 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | deep_nesting_16 | 1 | 0.3043 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | deep_nesting_256 | 1 | 0.3003 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | deep_nesting_64 | 1 | 0.3011 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | huge_numbers | 1 | 0.9999 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | long_composition_2 | 256 | 0.3833 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | long_formula_terms_128 | 1 | 0.4115 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | long_formula_terms_2048 | 1 | 0.4447 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | long_formula_terms_32 | 1 | 0.3729 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | long_formula_terms_512 | 1 | 0.4243 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | malformed_tex | 1 | 0.3333 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | rare_unicode | 1 | 0.9231 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | symbol_renaming | 239 | 0.4035 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | unknown_macro_unicode | 1 | 0.7708 |
| broad | synthetic_stress_test | hf_bpe_matched_fit | whitespace_removed | 187 | 0.3595 |
| broad | synthetic_stress_test | random | deep_nesting_16 | 1 | 0.3106 |
| broad | synthetic_stress_test | random | deep_nesting_256 | 1 | 0.3007 |
| broad | synthetic_stress_test | random | deep_nesting_64 | 1 | 0.3027 |
| broad | synthetic_stress_test | random | huge_numbers | 1 | 0.9999 |
| broad | synthetic_stress_test | random | long_composition_2 | 256 | 0.4048 |
| broad | synthetic_stress_test | random | long_formula_terms_128 | 1 | 0.4047 |
| broad | synthetic_stress_test | random | long_formula_terms_2048 | 1 | 0.4490 |
| broad | synthetic_stress_test | random | long_formula_terms_32 | 1 | 0.3888 |
| broad | synthetic_stress_test | random | long_formula_terms_512 | 1 | 0.4275 |
| broad | synthetic_stress_test | random | malformed_tex | 1 | 0.3333 |
| broad | synthetic_stress_test | random | rare_unicode | 1 | 0.9231 |
| broad | synthetic_stress_test | random | symbol_renaming | 239 | 0.4192 |
| broad | synthetic_stress_test | random | unknown_macro_unicode | 1 | 0.7917 |
| broad | synthetic_stress_test | random | whitespace_removed | 187 | 0.3837 |
| broad | synthetic_stress_test | reward_greedy | deep_nesting_16 | 1 | 0.3043 |
| broad | synthetic_stress_test | reward_greedy | deep_nesting_256 | 1 | 0.3003 |
| broad | synthetic_stress_test | reward_greedy | deep_nesting_64 | 1 | 0.3011 |
| broad | synthetic_stress_test | reward_greedy | huge_numbers | 1 | 0.9999 |
| broad | synthetic_stress_test | reward_greedy | long_composition_2 | 256 | 0.3730 |
| broad | synthetic_stress_test | reward_greedy | long_formula_terms_128 | 1 | 0.4072 |
| broad | synthetic_stress_test | reward_greedy | long_formula_terms_2048 | 1 | 0.4416 |
| broad | synthetic_stress_test | reward_greedy | long_formula_terms_32 | 1 | 0.3729 |
| broad | synthetic_stress_test | reward_greedy | long_formula_terms_512 | 1 | 0.4121 |
| broad | synthetic_stress_test | reward_greedy | malformed_tex | 1 | 0.3333 |
| broad | synthetic_stress_test | reward_greedy | rare_unicode | 1 | 0.9231 |
| broad | synthetic_stress_test | reward_greedy | symbol_renaming | 239 | 0.3926 |
| broad | synthetic_stress_test | reward_greedy | unknown_macro_unicode | 1 | 0.8333 |
| broad | synthetic_stress_test | reward_greedy | whitespace_removed | 187 | 0.3544 |
| broad | synthetic_stress_test | rl_composite | deep_nesting_16 | 1 | 0.3043 |
| broad | synthetic_stress_test | rl_composite | deep_nesting_256 | 1 | 0.3003 |
| broad | synthetic_stress_test | rl_composite | deep_nesting_64 | 1 | 0.3011 |
| broad | synthetic_stress_test | rl_composite | huge_numbers | 1 | 0.9999 |
| broad | synthetic_stress_test | rl_composite | long_composition_2 | 256 | 0.3741 |
| broad | synthetic_stress_test | rl_composite | long_formula_terms_128 | 1 | 0.4125 |
| broad | synthetic_stress_test | rl_composite | long_formula_terms_2048 | 1 | 0.4450 |
| broad | synthetic_stress_test | rl_composite | long_formula_terms_32 | 1 | 0.3749 |
| broad | synthetic_stress_test | rl_composite | long_formula_terms_512 | 1 | 0.4250 |
| broad | synthetic_stress_test | rl_composite | malformed_tex | 1 | 0.3333 |
| broad | synthetic_stress_test | rl_composite | rare_unicode | 1 | 0.9231 |
| broad | synthetic_stress_test | rl_composite | symbol_renaming | 239 | 0.3928 |
| broad | synthetic_stress_test | rl_composite | unknown_macro_unicode | 1 | 0.8125 |
| broad | synthetic_stress_test | rl_composite | whitespace_removed | 187 | 0.3554 |
| broad | synthetic_stress_test | rl_length_only | deep_nesting_16 | 1 | 0.3043 |
| broad | synthetic_stress_test | rl_length_only | deep_nesting_256 | 1 | 0.3003 |
| broad | synthetic_stress_test | rl_length_only | deep_nesting_64 | 1 | 0.3011 |
| broad | synthetic_stress_test | rl_length_only | huge_numbers | 1 | 0.9999 |
| broad | synthetic_stress_test | rl_length_only | long_composition_2 | 256 | 0.3910 |
| broad | synthetic_stress_test | rl_length_only | long_formula_terms_128 | 1 | 0.4072 |
| broad | synthetic_stress_test | rl_length_only | long_formula_terms_2048 | 1 | 0.4419 |
| broad | synthetic_stress_test | rl_length_only | long_formula_terms_32 | 1 | 0.3729 |
| broad | synthetic_stress_test | rl_length_only | long_formula_terms_512 | 1 | 0.4121 |
| broad | synthetic_stress_test | rl_length_only | malformed_tex | 1 | 0.3333 |
| broad | synthetic_stress_test | rl_length_only | rare_unicode | 1 | 0.9231 |
| broad | synthetic_stress_test | rl_length_only | symbol_renaming | 239 | 0.4084 |
| broad | synthetic_stress_test | rl_length_only | unknown_macro_unicode | 1 | 0.8125 |
| broad | synthetic_stress_test | rl_length_only | whitespace_removed | 187 | 0.3665 |
| broad | synthetic_stress_test | sp_bpe | deep_nesting_16 | 1 | 0.4099 |
| broad | synthetic_stress_test | sp_bpe | deep_nesting_256 | 1 | 0.4006 |
| broad | synthetic_stress_test | sp_bpe | deep_nesting_64 | 1 | 0.4025 |
| broad | synthetic_stress_test | sp_bpe | huge_numbers | 1 | 0.9999 |
| broad | synthetic_stress_test | sp_bpe | long_composition_2 | 256 | 0.4695 |
| broad | synthetic_stress_test | sp_bpe | long_formula_terms_128 | 1 | 0.4687 |
| broad | synthetic_stress_test | sp_bpe | long_formula_terms_2048 | 1 | 0.4989 |
| broad | synthetic_stress_test | sp_bpe | long_formula_terms_32 | 1 | 0.4367 |
| broad | synthetic_stress_test | sp_bpe | long_formula_terms_512 | 1 | 0.4711 |
| broad | synthetic_stress_test | sp_bpe | malformed_tex | 1 | 0.5556 |
| broad | synthetic_stress_test | sp_bpe | rare_unicode | 1 | 0.9231 |
| broad | synthetic_stress_test | sp_bpe | symbol_renaming | 239 | 0.4682 |
| broad | synthetic_stress_test | sp_bpe | unknown_macro_unicode | 1 | 0.7917 |
| broad | synthetic_stress_test | sp_bpe | whitespace_removed | 187 | 0.4731 |
| broad | synthetic_stress_test | sp_unigram | deep_nesting_16 | 1 | 0.4099 |
| broad | synthetic_stress_test | sp_unigram | deep_nesting_256 | 1 | 0.4006 |
| broad | synthetic_stress_test | sp_unigram | deep_nesting_64 | 1 | 0.4025 |
| broad | synthetic_stress_test | sp_unigram | huge_numbers | 1 | 0.9999 |
| broad | synthetic_stress_test | sp_unigram | long_composition_2 | 256 | 0.4905 |
| broad | synthetic_stress_test | sp_unigram | long_formula_terms_128 | 1 | 0.4783 |
| broad | synthetic_stress_test | sp_unigram | long_formula_terms_2048 | 1 | 0.5041 |
| broad | synthetic_stress_test | sp_unigram | long_formula_terms_32 | 1 | 0.4506 |
| broad | synthetic_stress_test | sp_unigram | long_formula_terms_512 | 1 | 0.5046 |
| broad | synthetic_stress_test | sp_unigram | malformed_tex | 1 | 0.5556 |
| broad | synthetic_stress_test | sp_unigram | rare_unicode | 1 | 0.9231 |
| broad | synthetic_stress_test | sp_unigram | symbol_renaming | 239 | 0.4886 |
| broad | synthetic_stress_test | sp_unigram | unknown_macro_unicode | 1 | 0.8542 |
| broad | synthetic_stress_test | sp_unigram | whitespace_removed | 187 | 0.4976 |
| prior_data_reference | extreme_domain_test | legacy_v3_rl | algebra | 327 | 0.4416 |
| prior_data_reference | extreme_domain_test | legacy_v3_rl | combinatorics | 472 | 0.4228 |
| prior_data_reference | extreme_domain_test | legacy_v3_rl | geometry_topology | 500 | 0.3836 |
| prior_data_reference | extreme_domain_test | legacy_v3_rl | logic | 46 | 0.4035 |
| prior_data_reference | extreme_domain_test | legacy_v3_rl | number_theory | 291 | 0.4928 |
| prior_data_reference | extreme_domain_test | legacy_v4_masked_rl | algebra | 327 | 0.4909 |
| prior_data_reference | extreme_domain_test | legacy_v4_masked_rl | combinatorics | 472 | 0.4386 |
| prior_data_reference | extreme_domain_test | legacy_v4_masked_rl | geometry_topology | 500 | 0.4240 |
| prior_data_reference | extreme_domain_test | legacy_v4_masked_rl | logic | 46 | 0.4035 |
| prior_data_reference | extreme_domain_test | legacy_v4_masked_rl | number_theory | 291 | 0.5351 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | algebra | 327 | 0.4416 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | analysis | 1568 | 0.3369 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | combinatorics | 472 | 0.4228 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | geometry_topology | 500 | 0.3836 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | linear_algebra | 13 | 0.4357 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | logic | 46 | 0.4035 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | mathematical_physics | 248 | 0.3824 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | number_theory | 291 | 0.4928 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | ode | 98 | 0.4126 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | optimization | 235 | 0.3535 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | pde | 31 | 0.3347 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | probability | 123 | 0.4284 |
| prior_data_reference | fresh_paper_test | legacy_v3_rl | statistics | 423 | 0.3024 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | algebra | 327 | 0.4909 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | analysis | 1568 | 0.3770 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | combinatorics | 472 | 0.4386 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | geometry_topology | 500 | 0.4240 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | linear_algebra | 13 | 0.4539 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | logic | 46 | 0.4035 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | mathematical_physics | 248 | 0.4046 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | number_theory | 291 | 0.5351 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | ode | 98 | 0.4346 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | optimization | 235 | 0.3937 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | pde | 31 | 0.3552 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | probability | 123 | 0.4694 |
| prior_data_reference | fresh_paper_test | legacy_v4_masked_rl | statistics | 423 | 0.3311 |
| reference | extreme_domain_test | tiktoken_cl100k_base | algebra | 327 | 0.5650 |
| reference | extreme_domain_test | tiktoken_cl100k_base | combinatorics | 472 | 0.5227 |
| reference | extreme_domain_test | tiktoken_cl100k_base | geometry_topology | 500 | 0.5482 |
| reference | extreme_domain_test | tiktoken_cl100k_base | logic | 46 | 0.5022 |
| reference | extreme_domain_test | tiktoken_cl100k_base | number_theory | 291 | 0.5676 |
| reference | extreme_domain_test | tiktoken_o200k_base | algebra | 327 | 0.5641 |
| reference | extreme_domain_test | tiktoken_o200k_base | combinatorics | 472 | 0.5272 |
| reference | extreme_domain_test | tiktoken_o200k_base | geometry_topology | 500 | 0.5488 |
| reference | extreme_domain_test | tiktoken_o200k_base | logic | 46 | 0.5145 |
| reference | extreme_domain_test | tiktoken_o200k_base | number_theory | 291 | 0.5692 |
| reference | fresh_paper_test | tiktoken_cl100k_base | algebra | 327 | 0.5650 |
| reference | fresh_paper_test | tiktoken_cl100k_base | analysis | 1568 | 0.5179 |
| reference | fresh_paper_test | tiktoken_cl100k_base | combinatorics | 472 | 0.5227 |
| reference | fresh_paper_test | tiktoken_cl100k_base | geometry_topology | 500 | 0.5482 |
| reference | fresh_paper_test | tiktoken_cl100k_base | linear_algebra | 13 | 0.5613 |
| reference | fresh_paper_test | tiktoken_cl100k_base | logic | 46 | 0.5022 |
| reference | fresh_paper_test | tiktoken_cl100k_base | mathematical_physics | 248 | 0.4987 |
| reference | fresh_paper_test | tiktoken_cl100k_base | number_theory | 291 | 0.5676 |
| reference | fresh_paper_test | tiktoken_cl100k_base | ode | 98 | 0.5667 |
| reference | fresh_paper_test | tiktoken_cl100k_base | optimization | 235 | 0.5367 |
| reference | fresh_paper_test | tiktoken_cl100k_base | pde | 31 | 0.4781 |
| reference | fresh_paper_test | tiktoken_cl100k_base | probability | 123 | 0.5746 |
| reference | fresh_paper_test | tiktoken_cl100k_base | statistics | 423 | 0.4917 |
| reference | fresh_paper_test | tiktoken_o200k_base | algebra | 327 | 0.5641 |
| reference | fresh_paper_test | tiktoken_o200k_base | analysis | 1568 | 0.5075 |
| reference | fresh_paper_test | tiktoken_o200k_base | combinatorics | 472 | 0.5272 |
| reference | fresh_paper_test | tiktoken_o200k_base | geometry_topology | 500 | 0.5488 |
| reference | fresh_paper_test | tiktoken_o200k_base | linear_algebra | 13 | 0.5628 |
| reference | fresh_paper_test | tiktoken_o200k_base | logic | 46 | 0.5145 |
| reference | fresh_paper_test | tiktoken_o200k_base | mathematical_physics | 248 | 0.5005 |
| reference | fresh_paper_test | tiktoken_o200k_base | number_theory | 291 | 0.5692 |
| reference | fresh_paper_test | tiktoken_o200k_base | ode | 98 | 0.5665 |
| reference | fresh_paper_test | tiktoken_o200k_base | optimization | 235 | 0.5335 |
| reference | fresh_paper_test | tiktoken_o200k_base | pde | 31 | 0.4756 |
| reference | fresh_paper_test | tiktoken_o200k_base | probability | 123 | 0.5685 |
| reference | fresh_paper_test | tiktoken_o200k_base | statistics | 423 | 0.4820 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | deep_nesting_16 | 1 | 0.5031 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | deep_nesting_256 | 1 | 0.5002 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | deep_nesting_64 | 1 | 0.5008 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | huge_numbers | 1 | 0.3336 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | long_composition_2 | 256 | 0.5129 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | long_formula_terms_128 | 1 | 0.5577 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | long_formula_terms_2048 | 1 | 0.5448 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | long_formula_terms_32 | 1 | 0.5743 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | long_formula_terms_512 | 1 | 0.5362 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | malformed_tex | 1 | 0.4444 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | rare_unicode | 1 | 0.6923 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | symbol_renaming | 239 | 0.5231 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | unknown_macro_unicode | 1 | 0.4583 |
| reference | synthetic_stress_test | tiktoken_cl100k_base | whitespace_removed | 187 | 0.5114 |
| reference | synthetic_stress_test | tiktoken_o200k_base | deep_nesting_16 | 1 | 0.5031 |
| reference | synthetic_stress_test | tiktoken_o200k_base | deep_nesting_256 | 1 | 0.5002 |
| reference | synthetic_stress_test | tiktoken_o200k_base | deep_nesting_64 | 1 | 0.5008 |
| reference | synthetic_stress_test | tiktoken_o200k_base | huge_numbers | 1 | 0.3336 |
| reference | synthetic_stress_test | tiktoken_o200k_base | long_composition_2 | 256 | 0.5053 |
| reference | synthetic_stress_test | tiktoken_o200k_base | long_formula_terms_128 | 1 | 0.4958 |
| reference | synthetic_stress_test | tiktoken_o200k_base | long_formula_terms_2048 | 1 | 0.4876 |
| reference | synthetic_stress_test | tiktoken_o200k_base | long_formula_terms_32 | 1 | 0.5105 |
| reference | synthetic_stress_test | tiktoken_o200k_base | long_formula_terms_512 | 1 | 0.4766 |
| reference | synthetic_stress_test | tiktoken_o200k_base | malformed_tex | 1 | 0.4444 |
| reference | synthetic_stress_test | tiktoken_o200k_base | rare_unicode | 1 | 0.6538 |
| reference | synthetic_stress_test | tiktoken_o200k_base | symbol_renaming | 239 | 0.5140 |
| reference | synthetic_stress_test | tiktoken_o200k_base | unknown_macro_unicode | 1 | 0.4375 |
| reference | synthetic_stress_test | tiktoken_o200k_base | whitespace_removed | 187 | 0.5005 |
| specialized | extreme_domain_test | frequency | algebra | 327 | 0.4766 |
| specialized | extreme_domain_test | frequency | combinatorics | 472 | 0.5459 |
| specialized | extreme_domain_test | frequency | geometry_topology | 500 | 0.4356 |
| specialized | extreme_domain_test | frequency | logic | 46 | 0.4502 |
| specialized | extreme_domain_test | frequency | number_theory | 291 | 0.5210 |
| specialized | extreme_domain_test | hf_bpe_fulltrain | algebra | 327 | 0.4751 |
| specialized | extreme_domain_test | hf_bpe_fulltrain | combinatorics | 472 | 0.5291 |
| specialized | extreme_domain_test | hf_bpe_fulltrain | geometry_topology | 500 | 0.4251 |
| specialized | extreme_domain_test | hf_bpe_fulltrain | logic | 46 | 0.4396 |
| specialized | extreme_domain_test | hf_bpe_fulltrain | number_theory | 291 | 0.5198 |
| specialized | extreme_domain_test | hf_bpe_matched_fit | algebra | 327 | 0.4766 |
| specialized | extreme_domain_test | hf_bpe_matched_fit | combinatorics | 472 | 0.5459 |
| specialized | extreme_domain_test | hf_bpe_matched_fit | geometry_topology | 500 | 0.4356 |
| specialized | extreme_domain_test | hf_bpe_matched_fit | logic | 46 | 0.4502 |
| specialized | extreme_domain_test | hf_bpe_matched_fit | number_theory | 291 | 0.5210 |
| specialized | extreme_domain_test | random | algebra | 327 | 0.4836 |
| specialized | extreme_domain_test | random | combinatorics | 472 | 0.5115 |
| specialized | extreme_domain_test | random | geometry_topology | 500 | 0.4234 |
| specialized | extreme_domain_test | random | logic | 46 | 0.4458 |
| specialized | extreme_domain_test | random | number_theory | 291 | 0.5127 |
| specialized | extreme_domain_test | reward_greedy | algebra | 327 | 0.4727 |
| specialized | extreme_domain_test | reward_greedy | combinatorics | 472 | 0.5126 |
| specialized | extreme_domain_test | reward_greedy | geometry_topology | 500 | 0.4218 |
| specialized | extreme_domain_test | reward_greedy | logic | 46 | 0.4335 |
| specialized | extreme_domain_test | reward_greedy | number_theory | 291 | 0.5171 |
| specialized | extreme_domain_test | rl_composite | algebra | 327 | 0.4732 |
| specialized | extreme_domain_test | rl_composite | combinatorics | 472 | 0.5113 |
| specialized | extreme_domain_test | rl_composite | geometry_topology | 500 | 0.4279 |
| specialized | extreme_domain_test | rl_composite | logic | 46 | 0.4414 |
| specialized | extreme_domain_test | rl_composite | number_theory | 291 | 0.5205 |
| specialized | extreme_domain_test | rl_length_only | algebra | 327 | 0.4705 |
| specialized | extreme_domain_test | rl_length_only | combinatorics | 472 | 0.5302 |
| specialized | extreme_domain_test | rl_length_only | geometry_topology | 500 | 0.4263 |
| specialized | extreme_domain_test | rl_length_only | logic | 46 | 0.4608 |
| specialized | extreme_domain_test | rl_length_only | number_theory | 291 | 0.5207 |
| specialized | extreme_domain_test | sp_bpe | algebra | 327 | 0.5149 |
| specialized | extreme_domain_test | sp_bpe | combinatorics | 472 | 0.5219 |
| specialized | extreme_domain_test | sp_bpe | geometry_topology | 500 | 0.4949 |
| specialized | extreme_domain_test | sp_bpe | logic | 46 | 0.5181 |
| specialized | extreme_domain_test | sp_bpe | number_theory | 291 | 0.5732 |
| specialized | extreme_domain_test | sp_unigram | algebra | 327 | 0.5181 |
| specialized | extreme_domain_test | sp_unigram | combinatorics | 472 | 0.4814 |
| specialized | extreme_domain_test | sp_unigram | geometry_topology | 500 | 0.4945 |
| specialized | extreme_domain_test | sp_unigram | logic | 46 | 0.5304 |
| specialized | extreme_domain_test | sp_unigram | number_theory | 291 | 0.5731 |
| specialized | synthetic_stress_test | frequency | deep_nesting_16 | 1 | 0.3043 |
| specialized | synthetic_stress_test | frequency | deep_nesting_256 | 1 | 0.3003 |
| specialized | synthetic_stress_test | frequency | deep_nesting_64 | 1 | 0.3011 |
| specialized | synthetic_stress_test | frequency | huge_numbers | 1 | 0.9999 |
| specialized | synthetic_stress_test | frequency | long_composition_2 | 256 | 0.4059 |
| specialized | synthetic_stress_test | frequency | long_formula_terms_128 | 1 | 0.4440 |
| specialized | synthetic_stress_test | frequency | long_formula_terms_2048 | 1 | 0.4742 |
| specialized | synthetic_stress_test | frequency | long_formula_terms_32 | 1 | 0.4088 |
| specialized | synthetic_stress_test | frequency | long_formula_terms_512 | 1 | 0.4549 |
| specialized | synthetic_stress_test | frequency | malformed_tex | 1 | 0.3333 |
| specialized | synthetic_stress_test | frequency | rare_unicode | 1 | 0.9615 |
| specialized | synthetic_stress_test | frequency | symbol_renaming | 239 | 0.4161 |
| specialized | synthetic_stress_test | frequency | unknown_macro_unicode | 1 | 0.8125 |
| specialized | synthetic_stress_test | frequency | whitespace_removed | 187 | 0.3760 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | deep_nesting_16 | 1 | 0.3043 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | deep_nesting_256 | 1 | 0.3003 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | deep_nesting_64 | 1 | 0.3011 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | huge_numbers | 1 | 0.9999 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | long_composition_2 | 256 | 0.4017 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | long_formula_terms_128 | 1 | 0.4377 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | long_formula_terms_2048 | 1 | 0.4688 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | long_formula_terms_32 | 1 | 0.4018 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | long_formula_terms_512 | 1 | 0.4489 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | malformed_tex | 1 | 0.3333 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | rare_unicode | 1 | 0.9615 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | symbol_renaming | 239 | 0.4154 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | unknown_macro_unicode | 1 | 0.7917 |
| specialized | synthetic_stress_test | hf_bpe_fulltrain | whitespace_removed | 187 | 0.3693 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | deep_nesting_16 | 1 | 0.3043 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | deep_nesting_256 | 1 | 0.3003 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | deep_nesting_64 | 1 | 0.3011 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | huge_numbers | 1 | 0.9999 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | long_composition_2 | 256 | 0.4059 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | long_formula_terms_128 | 1 | 0.4440 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | long_formula_terms_2048 | 1 | 0.4742 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | long_formula_terms_32 | 1 | 0.4088 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | long_formula_terms_512 | 1 | 0.4549 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | malformed_tex | 1 | 0.3333 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | rare_unicode | 1 | 0.9615 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | symbol_renaming | 239 | 0.4161 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | unknown_macro_unicode | 1 | 0.8125 |
| specialized | synthetic_stress_test | hf_bpe_matched_fit | whitespace_removed | 187 | 0.3760 |
| specialized | synthetic_stress_test | random | deep_nesting_16 | 1 | 0.3043 |
| specialized | synthetic_stress_test | random | deep_nesting_256 | 1 | 0.3003 |
| specialized | synthetic_stress_test | random | deep_nesting_64 | 1 | 0.3011 |
| specialized | synthetic_stress_test | random | huge_numbers | 1 | 0.9999 |
| specialized | synthetic_stress_test | random | long_composition_2 | 256 | 0.4089 |
| specialized | synthetic_stress_test | random | long_formula_terms_128 | 1 | 0.4590 |
| specialized | synthetic_stress_test | random | long_formula_terms_2048 | 1 | 0.5028 |
| specialized | synthetic_stress_test | random | long_formula_terms_32 | 1 | 0.4417 |
| specialized | synthetic_stress_test | random | long_formula_terms_512 | 1 | 0.4848 |
| specialized | synthetic_stress_test | random | malformed_tex | 1 | 0.3333 |
| specialized | synthetic_stress_test | random | rare_unicode | 1 | 0.9615 |
| specialized | synthetic_stress_test | random | symbol_renaming | 239 | 0.4147 |
| specialized | synthetic_stress_test | random | unknown_macro_unicode | 1 | 0.7917 |
| specialized | synthetic_stress_test | random | whitespace_removed | 187 | 0.3733 |
| specialized | synthetic_stress_test | reward_greedy | deep_nesting_16 | 1 | 0.3043 |
| specialized | synthetic_stress_test | reward_greedy | deep_nesting_256 | 1 | 0.3003 |
| specialized | synthetic_stress_test | reward_greedy | deep_nesting_64 | 1 | 0.3011 |
| specialized | synthetic_stress_test | reward_greedy | huge_numbers | 1 | 0.9999 |
| specialized | synthetic_stress_test | reward_greedy | long_composition_2 | 256 | 0.3985 |
| specialized | synthetic_stress_test | reward_greedy | long_formula_terms_128 | 1 | 0.4733 |
| specialized | synthetic_stress_test | reward_greedy | long_formula_terms_2048 | 1 | 0.5036 |
| specialized | synthetic_stress_test | reward_greedy | long_formula_terms_32 | 1 | 0.4556 |
| specialized | synthetic_stress_test | reward_greedy | long_formula_terms_512 | 1 | 0.4838 |
| specialized | synthetic_stress_test | reward_greedy | malformed_tex | 1 | 0.3333 |
| specialized | synthetic_stress_test | reward_greedy | rare_unicode | 1 | 0.9615 |
| specialized | synthetic_stress_test | reward_greedy | symbol_renaming | 239 | 0.4155 |
| specialized | synthetic_stress_test | reward_greedy | unknown_macro_unicode | 1 | 0.8125 |
| specialized | synthetic_stress_test | reward_greedy | whitespace_removed | 187 | 0.3745 |
| specialized | synthetic_stress_test | rl_composite | deep_nesting_16 | 1 | 0.3043 |
| specialized | synthetic_stress_test | rl_composite | deep_nesting_256 | 1 | 0.3003 |
| specialized | synthetic_stress_test | rl_composite | deep_nesting_64 | 1 | 0.3011 |
| specialized | synthetic_stress_test | rl_composite | huge_numbers | 1 | 0.9999 |
| specialized | synthetic_stress_test | rl_composite | long_composition_2 | 256 | 0.3968 |
| specialized | synthetic_stress_test | rl_composite | long_formula_terms_128 | 1 | 0.4478 |
| specialized | synthetic_stress_test | rl_composite | long_formula_terms_2048 | 1 | 0.4781 |
| specialized | synthetic_stress_test | rl_composite | long_formula_terms_32 | 1 | 0.4267 |
| specialized | synthetic_stress_test | rl_composite | long_formula_terms_512 | 1 | 0.4670 |
| specialized | synthetic_stress_test | rl_composite | malformed_tex | 1 | 0.3333 |
| specialized | synthetic_stress_test | rl_composite | rare_unicode | 1 | 0.9615 |
| specialized | synthetic_stress_test | rl_composite | symbol_renaming | 239 | 0.4124 |
| specialized | synthetic_stress_test | rl_composite | unknown_macro_unicode | 1 | 0.8125 |
| specialized | synthetic_stress_test | rl_composite | whitespace_removed | 187 | 0.3711 |
| specialized | synthetic_stress_test | rl_length_only | deep_nesting_16 | 1 | 0.3043 |
| specialized | synthetic_stress_test | rl_length_only | deep_nesting_256 | 1 | 0.3003 |
| specialized | synthetic_stress_test | rl_length_only | deep_nesting_64 | 1 | 0.3011 |
| specialized | synthetic_stress_test | rl_length_only | huge_numbers | 1 | 0.9999 |
| specialized | synthetic_stress_test | rl_length_only | long_composition_2 | 256 | 0.4019 |
| specialized | synthetic_stress_test | rl_length_only | long_formula_terms_128 | 1 | 0.4430 |
| specialized | synthetic_stress_test | rl_length_only | long_formula_terms_2048 | 1 | 0.4758 |
| specialized | synthetic_stress_test | rl_length_only | long_formula_terms_32 | 1 | 0.4247 |
| specialized | synthetic_stress_test | rl_length_only | long_formula_terms_512 | 1 | 0.4546 |
| specialized | synthetic_stress_test | rl_length_only | malformed_tex | 1 | 0.3333 |
| specialized | synthetic_stress_test | rl_length_only | rare_unicode | 1 | 0.9615 |
| specialized | synthetic_stress_test | rl_length_only | symbol_renaming | 239 | 0.4173 |
| specialized | synthetic_stress_test | rl_length_only | unknown_macro_unicode | 1 | 0.7708 |
| specialized | synthetic_stress_test | rl_length_only | whitespace_removed | 187 | 0.3759 |
| specialized | synthetic_stress_test | sp_bpe | deep_nesting_16 | 1 | 0.4099 |
| specialized | synthetic_stress_test | sp_bpe | deep_nesting_256 | 1 | 0.4006 |
| specialized | synthetic_stress_test | sp_bpe | deep_nesting_64 | 1 | 0.4025 |
| specialized | synthetic_stress_test | sp_bpe | huge_numbers | 1 | 0.9999 |
| specialized | synthetic_stress_test | sp_bpe | long_composition_2 | 256 | 0.4795 |
| specialized | synthetic_stress_test | sp_bpe | long_formula_terms_128 | 1 | 0.4633 |
| specialized | synthetic_stress_test | sp_bpe | long_formula_terms_2048 | 1 | 0.4958 |
| specialized | synthetic_stress_test | sp_bpe | long_formula_terms_32 | 1 | 0.4347 |
| specialized | synthetic_stress_test | sp_bpe | long_formula_terms_512 | 1 | 0.4684 |
| specialized | synthetic_stress_test | sp_bpe | malformed_tex | 1 | 0.5556 |
| specialized | synthetic_stress_test | sp_bpe | rare_unicode | 1 | 0.9615 |
| specialized | synthetic_stress_test | sp_bpe | symbol_renaming | 239 | 0.4796 |
| specialized | synthetic_stress_test | sp_bpe | unknown_macro_unicode | 1 | 0.7708 |
| specialized | synthetic_stress_test | sp_bpe | whitespace_removed | 187 | 0.4802 |
| specialized | synthetic_stress_test | sp_unigram | deep_nesting_16 | 1 | 0.4161 |
| specialized | synthetic_stress_test | sp_unigram | deep_nesting_256 | 1 | 0.4010 |
| specialized | synthetic_stress_test | sp_unigram | deep_nesting_64 | 1 | 0.4041 |
| specialized | synthetic_stress_test | sp_unigram | huge_numbers | 1 | 0.9999 |
| specialized | synthetic_stress_test | sp_unigram | long_composition_2 | 256 | 0.4764 |
| specialized | synthetic_stress_test | sp_unigram | long_formula_terms_128 | 1 | 0.4725 |
| specialized | synthetic_stress_test | sp_unigram | long_formula_terms_2048 | 1 | 0.5039 |
| specialized | synthetic_stress_test | sp_unigram | long_formula_terms_32 | 1 | 0.4207 |
| specialized | synthetic_stress_test | sp_unigram | long_formula_terms_512 | 1 | 0.4949 |
| specialized | synthetic_stress_test | sp_unigram | malformed_tex | 1 | 0.5556 |
| specialized | synthetic_stress_test | sp_unigram | rare_unicode | 1 | 0.9231 |
| specialized | synthetic_stress_test | sp_unigram | symbol_renaming | 239 | 0.4781 |
| specialized | synthetic_stress_test | sp_unigram | unknown_macro_unicode | 1 | 0.8542 |
| specialized | synthetic_stress_test | sp_unigram | whitespace_removed | 187 | 0.4782 |

## 限制

在新test只首次固定评估之后保留原始结果，不调整method让test更好。本文所有观察只是intrinsic分词泛化；未训练下游模型，不能称数学推理或生成OOD提高。
unigram bits/byte是经验token分布编码描述，不是LM BPB。tree metrics来自同parser，界外macro/损坏语法的结构质量不声称正确。新增语料自动来源筛选/质量门槛/拒收审计不等于全体数学专家审核。
legacy v3/v4在旧v2语料训练，fresh新论文可作为历史冻结模型参照；旧训练包含target领域，因此它们在extreme集合的结果不构成未见领域学习实验，也不是同训练数据公平排名。
旧成熟大词表tiktoken是contextual reference，不与训练1024的小词表作公平学习排名。所有失败、退化和没有改善指标应依据完整JSON报告。
