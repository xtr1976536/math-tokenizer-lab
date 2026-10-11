import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import read_jsonl

out=ROOT/sys.argv[1];p=json.loads((out/'protocol.json').read_text());s=json.loads((out/'selection.json').read_text());summary=json.loads((ROOT/'data/releases/v5/curated/summary.json').read_text());metrics=json.loads((out/'metrics.json').read_text());rows=read_jsonl(out/'per_formula.jsonl')
lines=['# 新一轮：RL词表、多目标奖励与泛化','',
 f"v5语料{summary['count']}条，新增{summary['new_count']}条、{summary['new_papers_downloaded']}篇固定许可arXiv源码。旧v2保持不变，新test hash先冻结。词表预算{p['capacity']}，正式三种子17/29/43。",
 'RL学习merge/table，推断使用官方Rust BPE。没有hard AST或手工命令清单，reward与候选统计仍需设计。复合奖励包含code cost/边界/rare/空白变体，不单纯token数；所有reward来自train内部paper holdout。',
 f"配置dev选择：{s['selections']}。两场景都冻结selection后才读fresh test，测试成绩未用来选择或修改配置。",'',
 '| 训练场景 | 测试集 | 方法 | token/公式 | tokens/byte | 命令完整率 | tree crossing | Rényi2 | 编码MiB/s | 无损率 |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
buckets=defaultdict(list)
for m in metrics:buckets[(m['scenario'],m['suite'],m['method'])].append(m)
for (scenario,suite,name),rr in sorted(buckets.items()):
 def avg(key):
  vals=[r[key] for r in rr if r[key] is not None];return float(np.mean(vals)) if vals else None
 def fmt(x):return f'{x:.4f}' if x is not None else 'N/A'
 lines.append('| '+' | '.join([scenario,suite,name]+[fmt(avg(k)) for k in ['tokens_per_formula','tokens_per_byte','command_integrity_rate','hybrid_tree_crossing_rate','renyi_efficiency_2','encode_mib_s','roundtrip_rate']])+' |')
lines+=['','## 泛化任务的含义','',
 'fresh_paper_test：新增从未参与旧实验的论文，按paper隔离，排除旧release和新train/dev相同词法模板。broad训练包含多个领域，因此这是新文档泛化，不是未见领域。',
 'extreme_domain_test：specialized只训练PDE/ODE/analysis，排除algebra/geometry/number theory/combinatorics/logic及cross-list源类别；再测新论文这些目标领域。这是本轮真正的领域迁移；与broad不同训练集/数据量，不能把差异全归因于域。',
 'synthetic_stress_test：heldout公式去空白/字母重命名/长式组合，以及2048项、深度256、rare Unicode/未知macro/巨大数字/损坏TeX。人工生成变体明确标synthetic，不能当自然科学公式或真值标签。',
 '', '## RL贡献与奖励消融','',
 'frequency/random/reward_greedy使用相同内部fit数据；HF matched-fit控制查明数据差异，HF full-train查实际成熟方法。reward greedy可能强于RL，不隐瞒；RL length-only是同路线奖励消融。',
 '奖励函数与policy是contextual合并，仍没有实现长时合并规划。统计features可以包含内部holdout各reward分量，不需要泛化到在线输入，因为policy只在离线学词表使用；最终官方BPE合并表是部署产物。',
 '当前全期望bandit gradient是确定性学习器：如果seed17/29/43给相同merge表，属于重复确定性运行，不是三个独立随机发现。',
 '这里“减少人介入”是删除手工命令词表/AST动作mask/逐式标注，不是删除设计者对reward、budget、feature和候选生成的选择。无法保证所有指标都优秀。',
 '', '## 速度解释','',
 'RLBPE和HF共用官方Rust实现、相同无regex预切分，速度差距主要来自学到的merge table/token分布。SentencePiece也是成熟原生实现，但算法不同；同数据、同vocab、同per-formula接口5次中位。',
 '不运行parser/policy意味着推断不再受Python结构前端拖慢。新算法是离线RL词表生成，不冒称同一v4路径策略的纯代码加速。',
 '', '## 统计区间','', '| 比较 | token/公式差值 | 95%paper bootstrap |','|---|---:|---|']
for key,r in json.loads((out/'pairwise.json').read_text()).items():lines.append(f'| {key} | {r["delta_tokens_per_formula"]:.4f} | {r["ci95_group_bootstrap"][0]:.4f}, {r["ci95_group_bootstrap"][1]:.4f} |')
lines+=['','## 按领域与stress类型','', '| 场景 | 集合 | 方法 | 子群 | 公式数 | token/byte |','|---|---|---|---|---:|---:|']
sub=defaultdict(list)
for r in rows:sub[(r['scenario'],r['suite'],r['method'],r.get('stress') or r['domain'])].append(r)
for (scenario,suite,method,group),rr in sorted(sub.items()):lines.append(f'| {scenario} | {suite} | {method} | {group} | {len({r["id"] for r in rr})} | {sum(r["tokens"] for r in rr)/sum(r["bytes"] for r in rr):.4f} |')
lines+=['','## 限制','',
 '在新test只首次固定评估之后保留原始结果，不调整method让test更好。本文所有观察只是intrinsic分词泛化；未训练下游模型，不能称数学推理或生成OOD提高。',
 'unigram bits/byte是经验token分布编码描述，不是LM BPB。tree metrics来自同parser，界外macro/损坏语法的结构质量不声称正确。新增语料自动来源筛选/质量门槛/拒收审计不等于全体数学专家审核。',
 'legacy v3/v4在旧v2语料训练，fresh新论文可作为历史冻结模型参照；旧训练包含target领域，因此它们在extreme集合的结果不构成未见领域学习实验，也不是同训练数据公平排名。',
 '旧成熟大词表tiktoken是contextual reference，不与训练1024的小词表作公平学习排名。所有失败、退化和没有改善指标应依据完整JSON报告。']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print('v5 report written')
