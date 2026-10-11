import json
from collections import defaultdict
import numpy as np
from math_tokenizer.corpus import ROOT

lines=['# 第四轮：硬约束与RL主线','',
 '语料与划分冻结，成熟基准未重训。核心RL仍是学参数的随机策略，训练后greedy/MAP解码；硬规则只定义合法动作，不替策略决定全部分词。',
 '当前解析器支持的源区间下partial crossing可由动作mask保证0，不能据此声称通用数学语义AST正确。未知宏/UTF8原子缺词表覆盖时显式byte escape，保证无损但不保证所有无限未知原子单ID。','',
 '| 预算 | 方法 | token/公式 | command完整率 | tree crossing | 编码MiB/s |','|---|---|---:|---:|---:|---:|']
for cap in (512,1024):
 out=ROOT/f'results/v4_{cap}';m=json.loads((out/'metrics.json').read_text());groups=defaultdict(list)
 for r in m:groups[r['method']].append(r)
 for name,rr in groups.items():
  avg=lambda key:float(np.mean([r[key] for r in rr]));lines.append(f'| {cap} | {name} | {avg("tokens_per_formula"):.4f} | {avg("command_integrity_rate"):.4%} | {avg("hybrid_tree_crossing_rate"):.4%} | {avg("encode_mib_s"):.4f} |')
 selection=json.loads((out/'selection.json').read_text());lines+=['',f'{cap}预算dev选择：{selection}']
 lines+=['','该预算配对区间：','', '| 比较 | token/公式差值 | 95%组bootstrap |','|---|---:|---|']
 for name,r in json.loads((out/'pairwise.json').read_text()).items():lines.append(f'| {name} | {r["delta_tokens_per_formula"]:.4f} | {r["ci95_group_bootstrap"][0]:.4f}, {r["ci95_group_bootstrap"][1]:.4f} |')
lines+=['','## 改进与代价','',
 '跨树归零来自hard合法区间mask；完整控制序列/codepoint保留来自硬原子边界，RL在剩余合法选择中学习，不把约束带来的指标归零冒称RL自己发现数学语法。v3-mask-only控制直接保留旧词表/旧策略再加mask，区分约束和重新训练的收益。',
 '全期望on-policy梯度等价于对REINFORCE全部轨迹取期望，在可枚举DAG上计算，不使用shortest-path teacher或黄金分词标签。expected-cost训练曲线保存，零参数训练起点提供学习证据。',
 'same-vocab shortest是压缩下界控制；策略无法保证优于这个下界。uniform-zero策略在greedy解码中由于tie可能选byte，不能拿它作为唯一强基准。',
 '硬树+完整命令可能使序列变长，尤其512词表用于保留全部命令后剩余合并预算少。若数字和符号完整更好但压缩下降，报告真实取舍，不宣称所有指标大幅改善。',
 '语法规则来自格式合法性；命令原子从train语料自动提取，无人工逐式编辑。配置仍由dev选择，这不等于零人工参与。',
 'RL train_seconds仅计策略学习，词表训练另记，不能误当总训练时间。每预算123条dev记录和14条test方法/seed记录，所有候选保留。',
 '速度包括真实结构解析与mask构建，每轮冷缓存；原生首次编译不包含稳态encode。含gater的原论文端到端架构未复现，独立tokenizer限制保持。',
 '1024预算matched-initial控制比训练后policy-MAP略短约0.0025 token/公式：说明随机策略期望回报改善未必改善确定性MAP分词，RL独立收益尚未证明，未来要对齐训练/推断目标。',
 '本轮硬约束使速度比第三轮略慢，不能声称本轮继续提速；保留前轮原生kernel的加速基础。',
 '源资料见docs/constraints_v4_research.md；固定语料的tokenizer-only结果没有下游性能结论，test是此前开发暴露集合。']
(ROOT/'docs/constraints_v4_findings.md').write_text('\n'.join(lines)+'\n')
