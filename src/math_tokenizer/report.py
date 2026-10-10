import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .evaluate import read_jsonl

def report(out):
    metrics=json.loads((out/'metrics.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    by=defaultdict(list)
    for row in metrics:by[row['method']].append(row)
    lines=['# 独立数学分词器：首轮本地实验', '',
           '这是 tokenizer-only pilot。奖励是训练内部留出数据的压缩增益，未训练或拼接任何语言模型，不能推断下游性能。', '',
           f"训练公式 {manifest['train_count']} 条，测试 {manifest['test_count']} 条，总词表预算 {manifest['capacity']}；自研方法随机种子 {manifest['seeds']}。",
           '所有方法使用同一测试 ID、原始 LaTeX、无截断、无特殊 token，逐条检查无损解码。AST 不支持的公式保留在共同测试集中，自研结构方法回退为 UTF-8 字节。', '',
           '| 方法 | 比较组 | 实际词表 | token/公式（种子均值） | bytes/token | 无损率 | 编码 MiB/s | 训练秒 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,rs in by.items():
        mean=lambda k:float(np.mean([r[k] for r in rs]))
        lines.append(f"| {name} | {rs[0]['comparison_group']} | {int(mean('vocab_actual'))} | {mean('tokens_per_formula'):.3f} | {mean('bytes_per_token'):.3f} | {mean('roundtrip_rate'):.4%} | {mean('encode_mib_s'):.3f} | {mean('train_seconds'):.3f} |")
    lines+=['','实际词表大小必须与预算同时解读：SentencePiece 需要未知符号和字节回退项；预训练参考具有不同规模词表及外部训练语料，不能用这张表宣称其训练方法优于或弱于自研方法。',
            'RL 使用 80% 训练公式收集合并候选、20% 训练公式学习奖励；额外列出的 *_rl_fit_control 使用相同 80% 候选拟合子集，供判断策略学习效果。全部训练数据算法比较与内部消融比较都保留。',
            '', '## 分领域／来源结果（自研种子均值和参考单次结果）','',
            '| 方法 | 领域 | 来源 | 公式数 | token/公式 | 无损率 |','|---|---|---|---:|---:|---:|']
    rows=read_jsonl(out/'per_formula.jsonl');buckets=defaultdict(list)
    for r in rows:buckets[(r['method'],r['domain'],r['origin'])].append(r)
    for (method,domain,origin),rs in sorted(buckets.items()):
        n=len({r['id'] for r in rs})
        lines.append(f"| {method} | {domain} | {origin} | {n} | {np.mean([r['tokens'] for r in rs]):.3f} | {np.mean([r['roundtrip'] for r in rs]):.4%} |")
    lines+=['','## 配对组 bootstrap','',
            '按文档／公式家族重采样 1000 次；差值为 A-B 的 token/公式，负值有利于 A。这是给定训练种子的条件区间，不是训练随机性区间。', '',
            '| 比较 | 差值 | 95% 区间 | 独立组数 |','|---|---:|---|---:|']
    pair=json.loads((out/'pairwise.json').read_text())
    for name,r in pair.items():lines.append(f"| {name} | {r['delta_tokens_per_formula']:.3f} | {r['ci95_group_bootstrap'][0]:.3f}, {r['ci95_group_bootstrap'][1]:.3f} | {r['groups']} |")
    lines+=['','## 解释与限制','',
            '1. 压缩、无损、解析覆盖率和速度是独立分词器指标，不等同于语言模型 BPB、数学推理或 PDE 识别准确率。',
            '2. AST 是保留源位置的受限语义表达式树。未知宏、矩阵、求和／积分绑定和复杂系统尚未完整解析；回退比例可能主导压缩结果，必须先提高覆盖率再做方法结论。',
            '3. RL 是离线 contextual merge policy，用 REINFORCE 学习候选合并选择。它不是 Dauncey & Wattenhofer 的端到端语言模型边界学习复现，也未实现其 early-exit 语言模型奖励。',
            '4. human 表示原始数学结构的人工作者证据，LaTeX 的自动映射另有标签；unknown 不能当成人工，OpenAI 论文保守标为 possible_mixed。AI synthetic 是本助手编写模板的机械扩展，不能当作独立模型大样本。',
            '5. 相同词表预算、相同测试协议不意味着计算预算相同；速度包括真实编码及结构解析，Python 和 Rust/C++ 的差异不能归因于算法本身。',
            '6. 同文档隔离导致某些领域尚无独立测试文档；没有测试数据的类别不提供结果。分组 bootstrap 中部分组只有少量数据，置信区间不能外推到未收集领域。',
            '', '原始逐公式结果、运行环境、输入 ID、数据／模型哈希与配置保存在本目录，可独立复核。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
