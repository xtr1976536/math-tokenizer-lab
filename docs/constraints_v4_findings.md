# 第四轮：硬约束与RL主线

语料与划分冻结，成熟基准未重训。核心RL仍是学参数的随机策略，训练后greedy/MAP解码；硬规则只定义合法动作，不替策略决定全部分词。
当前解析器支持的源区间下partial crossing可由动作mask保证0，不能据此声称通用数学语义AST正确。未知宏/UTF8原子缺词表覆盖时显式byte escape，保证无损但不保证所有无限未知原子单ID。

| 预算 | 方法 | token/公式 | command完整率 | tree crossing | 编码MiB/s |
|---|---|---:|---:|---:|---:|
| 512 | ast_v4 | 19.4264 | 99.5221% | 0.0000% | 1.0847 |
| 512 | ast_v3 | 18.5090 | 96.4953% | 0.0000% | 1.1847 |
| 512 | masked_zero_policy | 22.9520 | 99.5221% | 0.0000% | 1.0951 |
| 512 | masked_shortest_control | 19.4264 | 99.5221% | 0.0000% | 1.1071 |
| 512 | rl_v3 | 17.2916 | 86.8307% | 13.1681% | 1.1578 |
| 512 | v3_mask_only_control | 20.0418 | 86.8307% | 0.0000% | 1.0819 |
| 512 | rl_v4 | 19.4269 | 99.5221% | 0.0000% | 1.0980 |
| 512 | masked_matched_initial_policy | 22.9520 | 99.5221% | 0.0000% | 1.0926 |

512预算dev选择：{'ast_atoms': 256, 'rl_atoms': 256, 'rl_config': 'expected12', 'rl_mode': 'greedy', 'dev_scores': {'ast': {'macro_tokens_per_byte': 0.43459700421721204, 'tokens_per_formula': 18.980413105413106, 'command_integrity': 0.9993812646949635}, 'rl_macro_tokens_per_byte': 0.43460142496644066}, 'test_used_for_selection': False}

该预算配对区间：

| 比较 | token/公式差值 | 95%组bootstrap |
|---|---:|---|
| rl_v4-vs-rl_v3/17 | 2.1361 | 1.5946, 2.9388 |
| rl_v4-vs-masked_zero_policy/17 | -3.5251 | -5.0280, -2.6107 |
| rl_v4-vs-masked_shortest_control/17 | 0.0005 | 0.0000, 0.0018 |
| rl_v4-vs-v3_mask_only_control/17 | -0.6186 | -1.2791, -0.1359 |
| rl_v4-vs-rl_v3/29 | 2.2485 | 1.6714, 3.1221 |
| rl_v4-vs-masked_zero_policy/29 | -3.5251 | -5.0280, -2.6107 |
| rl_v4-vs-masked_shortest_control/29 | 0.0005 | 0.0000, 0.0018 |
| rl_v4-vs-v3_mask_only_control/29 | -0.6133 | -1.2742, -0.1333 |
| rl_v4-vs-rl_v3/43 | 2.0212 | 1.5186, 2.7801 |
| rl_v4-vs-masked_zero_policy/43 | -3.5251 | -5.0280, -2.6107 |
| rl_v4-vs-masked_shortest_control/43 | 0.0005 | 0.0000, 0.0018 |
| rl_v4-vs-v3_mask_only_control/43 | -0.6130 | -1.2738, -0.1333 |
| ast_v4-vs-v3 | 0.9174 | 0.5466, 1.2833 |
| rl_v4-vs-matched_initial/17 | -3.5251 | -5.0280, -2.6107 |
| rl_v4-vs-matched_initial/29 | -3.5251 | -5.0280, -2.6107 |
| rl_v4-vs-matched_initial/43 | -3.5251 | -5.0280, -2.6107 |
| 1024 | ast_v4 | 14.9227 | 99.5221% | 0.0000% | 1.0841 |
| 1024 | ast_v3 | 14.9752 | 96.4953% | 0.0000% | 1.1633 |
| 1024 | masked_zero_policy | 19.3428 | 99.5221% | 0.0000% | 1.0696 |
| 1024 | masked_shortest_control | 14.9227 | 99.5221% | 0.0000% | 1.0872 |
| 1024 | rl_v3 | 13.9361 | 96.4953% | 18.1786% | 1.1484 |
| 1024 | v3_mask_only_control | 17.1295 | 96.4953% | 0.0000% | 1.0745 |
| 1024 | rl_v4 | 14.9462 | 99.5221% | 0.0000% | 1.0718 |
| 1024 | masked_matched_initial_policy | 14.9437 | 99.5221% | 0.0000% | 1.0793 |

1024预算dev选择：{'ast_atoms': 256, 'rl_atoms': 256, 'rl_config': 'expected24_prior', 'rl_mode': 'policy_map', 'dev_scores': {'ast': {'macro_tokens_per_byte': 0.26648878886350363, 'tokens_per_formula': 14.133190883190883, 'command_integrity': 0.9993812646949635}, 'rl_macro_tokens_per_byte': 0.2672032025681046}, 'test_used_for_selection': False}

该预算配对区间：

| 比较 | token/公式差值 | 95%组bootstrap |
|---|---:|---|
| rl_v4-vs-rl_v3/17 | 1.0104 | 0.7413, 1.4166 |
| rl_v4-vs-masked_zero_policy/17 | -4.3966 | -5.4529, -3.6742 |
| rl_v4-vs-masked_shortest_control/17 | 0.0235 | 0.0145, 0.0350 |
| rl_v4-vs-v3_mask_only_control/17 | -2.1834 | -3.2629, -1.5241 |
| rl_v4-vs-rl_v3/29 | 1.0106 | 0.7415, 1.4166 |
| rl_v4-vs-masked_zero_policy/29 | -4.3966 | -5.4529, -3.6742 |
| rl_v4-vs-masked_shortest_control/29 | 0.0235 | 0.0145, 0.0350 |
| rl_v4-vs-v3_mask_only_control/29 | -2.1831 | -3.2624, -1.5239 |
| rl_v4-vs-rl_v3/43 | 1.0093 | 0.7404, 1.4145 |
| rl_v4-vs-masked_zero_policy/43 | -4.3966 | -5.4529, -3.6742 |
| rl_v4-vs-masked_shortest_control/43 | 0.0235 | 0.0145, 0.0350 |
| rl_v4-vs-v3_mask_only_control/43 | -2.1834 | -3.2629, -1.5241 |
| ast_v4-vs-v3 | -0.0525 | -0.2782, 0.1361 |
| rl_v4-vs-matched_initial/17 | 0.0025 | 0.0000, 0.0075 |
| rl_v4-vs-matched_initial/29 | 0.0025 | 0.0000, 0.0075 |
| rl_v4-vs-matched_initial/43 | 0.0025 | 0.0000, 0.0075 |

## 改进与代价

跨树归零来自hard合法区间mask；完整控制序列/codepoint保留来自硬原子边界，RL在剩余合法选择中学习，不把约束带来的指标归零冒称RL自己发现数学语法。v3-mask-only控制直接保留旧词表/旧策略再加mask，区分约束和重新训练的收益。
全期望on-policy梯度等价于对REINFORCE全部轨迹取期望，在可枚举DAG上计算，不使用shortest-path teacher或黄金分词标签。expected-cost训练曲线保存，零参数训练起点提供学习证据。
same-vocab shortest是压缩下界控制；策略无法保证优于这个下界。uniform-zero策略在greedy解码中由于tie可能选byte，不能拿它作为唯一强基准。
硬树+完整命令可能使序列变长，尤其512词表用于保留全部命令后剩余合并预算少。若数字和符号完整更好但压缩下降，报告真实取舍，不宣称所有指标大幅改善。
语法规则来自格式合法性；命令原子从train语料自动提取，无人工逐式编辑。配置仍由dev选择，这不等于零人工参与。
RL train_seconds仅计策略学习，词表训练另记，不能误当总训练时间。每预算123条dev记录和14条test方法/seed记录，所有候选保留。
速度包括真实结构解析与mask构建，每轮冷缓存；原生首次编译不包含稳态encode。含gater的原论文端到端架构未复现，独立tokenizer限制保持。
1024预算matched-initial控制比训练后policy-MAP略短约0.0025 token/公式：说明随机策略期望回报改善未必改善确定性MAP分词，RL独立收益尚未证明，未来要对齐训练/推断目标。
本轮硬约束使速度比第三轮略慢，不能声称本轮继续提速；保留前轮原生kernel的加速基础。
源资料见docs/constraints_v4_research.md；固定语料的tokenizer-only结果没有下游性能结论，test是此前开发暴露集合。
