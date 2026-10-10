import json
from collections import defaultdict
import numpy as np
from .evaluate import read_jsonl

def report(out):
    protocol=json.loads((out/'protocol.json').read_text());metrics=json.loads((out/'metrics.json').read_text());selection=json.loads((out/'selection.json').read_text())
    by=defaultdict(list)
    for m in metrics:by[m['method']].append(m)
    lines=['# 第二版独立数学分词器实验','',
        f"训练{len(protocol['train_ids'])}条、验证{len(protocol['dev_ids'])}条、测试{len(protocol['test_ids'])}条；训练词表预算{protocol['capacity']}；随机种子{protocol['seeds']}。",
        'RL完整路径奖励只使用训练公式；配置仅按验证文档宏平均token/byte选择，测试在selection.json冻结后统一调用。不同模型都用相同原始LaTeX，无截断、无特殊token。',
        '结构树分 semantic_ast 与 latex_syntax_tree，后者是保留命令和嵌套的语法树，不能称通用语义AST。', '',
        '| 方法 | 比较组 | token/公式 | bytes/token | 词表利用率 | Rényi效率 α=2 | 命令完整率 | 跨树比例 | 无损率 | 编码MiB/s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,rs in by.items():
        def mean(key):return np.mean([r[key] for r in rs if r[key] is not None]) if any(r[key] is not None for r in rs) else None
        def fmt(x):return f'{x:.4f}' if x is not None else 'N/A'
        renyi=np.mean([r['renyi_efficiency']['2.0'] for r in rs])
        lines.append('| '+ ' | '.join([name,rs[0]['comparison_group'],fmt(mean('tokens_per_formula')),fmt(mean('bytes_per_token')),fmt(mean('vocab_utilization')),fmt(renyi),fmt(mean('command_integrity_rate')),fmt(mean('hybrid_tree_crossing_rate')),fmt(mean('roundtrip_rate')),fmt(mean('encode_mib_s'))])+' |')
    lines+=['','## 验证集消融与选择','',f"固定选择：{selection['selected']}",'',
        '| 分支 | 配置 | 两种子验证文档宏平均token/byte |','|---|---|---:|']
    for arm,scores in selection['dev_config_mean_scores'].items():
        for name,score in sorted(scores.items()):lines.append(f'| {arm} | {name} | {score:.6f} |')
    lines+=['','所有配置、两个种子、epoch训练损失、梯度样本方差均保存在dev_ablations.json；没有按测试选择最好的配置。no_rl与shortest_path控制使用同一BPE词表：前者测试无策略学习时的长度先验，后者是这个固定词表与相同约束下的最少token路径参照。',
        '', '## 按来源、领域、复杂度分层','',
        '| 方法 | 领域 | 来源 | 复杂度 | 独立公式数 | token/公式 | token/byte |','|---|---|---|---|---:|---:|---:|']
    rows=read_jsonl(out/'per_formula.jsonl');buckets=defaultdict(list)
    for r in rows:buckets[(r['method'],r['domain'],r['origin'],r['complexity'])].append(r)
    for (method,domain,origin,complexity),rs in sorted(buckets.items()):
        lines.append(f"| {method} | {domain} | {origin} | {complexity} | {len({r['id'] for r in rs})} | {np.mean([r['tokens'] for r in rs]):.4f} | {sum(r['tokens'] for r in rs)/sum(r['bytes'] for r in rs):.4f} |")
    lines+=['','## 组配对置信区间','', '| 比较 | token/公式差值 | 文档/家族bootstrap95%区间 | 组数 |','|---|---:|---|---:|']
    for key,r in json.loads((out/'pairwise.json').read_text()).items():lines.append(f"| {key} | {r['delta_tokens_per_formula']:.4f} | {r['ci95_group_bootstrap'][0]:.4f}, {r['ci95_group_bootstrap'][1]:.4f} | {r['groups']} |")
    lines+=['','## 指标的用途与限制','',
      '指标细节、方向、相关论文与官方实现见docs/intrinsic_metrics_v2.md。Rényi效率、熵、empirical unigram bits/byte都只是tokenizer-only描述；经验unigram编码成本不是语言模型BPB。',
      '固定预训练词表单列，不用不同词表和外部语料得出的压缩率作为公平训练算法胜负。算法使用相同数据与vocab上限；实际训练算力有差别，秒数和策略训练样本量透明记录。',
      '测试公式全部参与，每条有结果；没有因结构解析失败移除困难公式。结构解析可以缓存用于指标分析，但每个方法每轮速度测试前清缓存，比较的是完整解析+编码成本。',
      '2020年前论文按用户批准的历史来源规则标human，证据强度另存，不声称逐作者无自动化辅助认证。Fungrim来源不确定性保留，机械符号转LaTeX与LLM生成区分记录。',
      '本版未训练下游模型，不报告推理、PDE识别或语言模型性能改善。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    import sys
    from pathlib import Path
    report(Path(sys.argv[1]))
