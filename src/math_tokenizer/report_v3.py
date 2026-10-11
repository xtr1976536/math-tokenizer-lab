import json
from collections import defaultdict
import numpy as np
from .evaluate import read_jsonl

def report(out):
    protocol=json.loads((out/'protocol.json').read_text());selection=json.loads((out/'selection.json').read_text());metrics=json.loads((out/'metrics.json').read_text());by=defaultdict(list)
    for m in metrics:by[m['method']].append(m)
    lines=['# 第三版优化：固定语料、无新基准','',
        f"词表预算{protocol['capacity']}，共同测试{len(protocol['test_ids'])}条。数学语料和split哈希冻结；仅重测现有自研方法以取得同进程/同输入的速度比较。没有新训练成熟基准。",
        f"配置由dev选择：{selection['ast']}、{selection['rl_byte']}。原生代码首次编译不计入稳态encode；每轮清结构缓存，实际结构解析计时。",
        '', '| 方法 | token/公式 | 命令完整率 | 数字完整率 | lexeme保留率 | 跨树比例 | Rényi α=2 | 编码MiB/s |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,rs in by.items():
        def avg(key):return float(np.mean([r[key] for r in rs]))
        lines.append(f"| {name} | {avg('tokens_per_formula'):.4f} | {avg('command_integrity_rate'):.4f} | {avg('number_integrity_rate'):.4f} | {avg('lexeme_preservation_rate'):.4f} | {avg('hybrid_tree_crossing_rate'):.4f} | {np.mean([r['renyi_efficiency']['2.0'] for r in rs]):.4f} | {avg('encode_mib_s'):.4f} |")
    lines+=['','## 配对文档置信区间','', '| 比较 | token/公式差值 | 95%组bootstrap |','|---|---:|---|']
    for name,r in json.loads((out/'pairwise.json').read_text()).items():lines.append(f"| {name} | {r['delta_tokens_per_formula']:.4f} | {r['ci95_group_bootstrap'][0]:.4f}, {r['ci95_group_bootstrap'][1]:.4f} |")
    lines+=['','## 解释','',
        'AST-v3仍先构建结构，训练词表时预留训练集常见LaTeX命令原子，之后做受源跨度约束的BPE。推断改为受结构约束的最短路径，因此不是纯ranked BPE；词表训练与推断改动的收益不能都归因于AST本身。',
        'RL-v3复用原始REINFORCE路径策略，命令原子种子+原生推断+验证选择的soft tree penalty；RL已训练、概率和参数保存。相同新词表no-RL shortest控制用于拆分词表/搜索与策略学习收益。',
        'exact native模式与v2输出一致，用于单独验证速度优化。原生kernel用于所有v3模型，Python语法结构解析保持一致；端到端速度仍受parser影响。',
        '更高命令完整性并不必然使Rényi熵、词表利用率等所有指标同向改善，完整JSON保留指标取舍，不把未改善指标隐藏。',
        '测试集是此前已查看的开发test，配置仍只按dev选择；本轮不得称最终未见测试结论。模型质量提升与代码实现速度提升分别解释。']
    rows=read_jsonl(out/'per_formula.jsonl');buckets=defaultdict(list)
    for r in rows:buckets[(r['method'],r['complexity'],r['domain'])].append(r)
    lines+=['','## 分层','', '| 方法 | 复杂度 | 领域 | 公式数 | token/公式 |','|---|---|---|---:|---:|']
    for (method,complexity,domain),rr in sorted(buckets.items()):lines.append(f"| {method} | {complexity} | {domain} | {len({r['id'] for r in rr})} | {np.mean([r['tokens'] for r in rr]):.4f} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n')
