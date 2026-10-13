"""Recompute exploratory stage-2 success gates; never promote on one proxy."""
import gzip,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.signal_stage2 import SignalTokenizer
from math_tokenizer.predictive_stage2 import PredictiveTokenizer
from math_tokenizer.learned_units_stage2 import LearnedUnitBPE


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    folder=ROOT/'experiments/stage2';runs={};review=[]
    for p in sorted(folder.glob('pilot*')):
        # Later neural/deployment experiments have their own manifest schema
        # and are reviewed by review_stage2_continuation.py.
        if not (p/'completion.json').exists() or not (p/'run_manifest.json').exists():continue
        manifest=json.loads((p/'run_manifest.json').read_text());freeze=json.loads((p/'freeze.json').read_text())
        for path,digest in manifest['source_sha256'].items():assert sha(p/'source_snapshot'/path)==digest
        for path,digest in freeze['model_sha256'].items():assert sha(p/path)==digest
        metrics=json.loads((p/'metrics.json').read_text());by={(m['method'],m['suite']):m for m in metrics}
        audits=json.loads((p/'audits.json').read_text());assert len(audits)>=3 and all(r['passed'] for r in audits)
        gates=[]
        for m in metrics:
            if m['suite']=='dev' or m['method'] in ('byte_bpe','math_atom_bpe','boundary_ast','unigram','qwen25','llama3','o200k'):continue
            ref=by[('boundary_ast',m['suite'])];deterioration=m['document_macro_tokens_per_byte']/ref['document_macro_tokens_per_byte']-1
            failures=[]
            if deterioration>.01:failures.append('compression')
            for key in ('command_integrity','number_integrity','independent_scope_boundary_recall','lexical_boundary_f1'):
                if m[key] is not None and ref[key] is not None and m[key]<ref[key]-.01:failures.append(key)
            if m['independent_tree_crossing']>ref['independent_tree_crossing']+.01:failures.append('tree_crossing')
            if m['encode_mib_s']<.9*ref['encode_mib_s']:failures.append('speed')
            if m['roundtrip_rate']!=1 or m['payload_fidelity_rate']!=1:failures.append('fidelity')
            substantial=(deterioration<=-.05 or m['encode_mib_s']>=1.2*ref['encode_mib_s'] or
                m['independent_scope_boundary_recall']>=ref['independent_scope_boundary_recall']+.05 or
                m['independent_tree_crossing']<=ref['independent_tree_crossing']-.05)
            gates.append({'method':m['method'],'suite':m['suite'],'failures':failures,'substantial_gain':substantial,
                          'deployment_gate_passed':not failures and substantial,
                          'token_byte_regression_vs_ast':deterioration,'encode_ratio_vs_ast':m['encode_mib_s']/ref['encode_mib_s']})
        comparisons=json.loads((p/'router_comparisons.json').read_text());raw=json.loads((p/'router_results.json').read_text())
        learning={}
        for control in ('random_search','cem_search','no_update'):
            cc=[x for x in comparisons if x['control']==control]
            learning[control]={'relative_bpb_improvement_percent':[100*x['relative_byte_loss_improvement'] for x in cc],
                'all_seeds_above_1_percent':all(x['relative_byte_loss_improvement']>=.01 for x in cc),
                'all_exploratory_paper_intervals_negative':all(x['paper_cluster_ci95'][1]<0 for x in cc)}
        # Independent full row aggregate replay and exact public encoding identity.
        grouped=defaultdict(list)
        with gzip.open(p/'per_formula.jsonl.gz','rt') as f:
            for line in f:r=json.loads(line);grouped[(r['method'],r['suite'])].append(r)
        for k,rr in grouped.items():
            m=by[k];assert len(rr)==m['formulas'] and sum(r['tokens'] for r in rr)==m['tokens']
            assert all(r['roundtrip'] and r['payload_exact'] for r in rr)
        zero_digests={};duplicates=[]
        for path in sorted((p/'models').glob('*.npz')):
            if path.stem in ('byte_signals',) or path.stem.endswith('_router'):continue
            with np.load(path,allow_pickle=False) as d:predictive='bits' in d.files
            model=(PredictiveTokenizer if predictive else SignalTokenizer).load(path)
            digest=hashlib.sha256('\n'.join(model.vocab[i].hex() for text in ('abc',r'\frac{x+y}{z}',r'\unknown{α😊}') for i in model.encode(text)).encode()).hexdigest()
            if np.count_nonzero(model.costs)==0 and not predictive:duplicates.append(path.stem)
            zero_digests[path.stem]=digest
        counts=len(grouped);total=sum(map(len,grouped.values()))
        review.append({'pilot':p.name,'source_snapshots_match':True,'frozen_models_match':True,'independent_rows_checked':total,
                       'metric_records':counts,'zero_signal_deployed_variants':duplicates})
        runs[p.name]={'deployment_gates':gates,'learning_proxy':learning,'models':len({m['method'] for m in metrics}),
                       'metrics':len(metrics),'rows':total,'all_fidelity_passed':True,
                       'successful_stage2_models':sorted({g['method'] for g in gates if all(x['deployment_gate_passed'] for x in gates if x['method']==g['method'])})}
    result={'status':'exploratory; no stage-2 success or production promotion',
            'pilots':runs,'independent_replay':review,'confirmed_success':False,
            'limitations':['same internal scopes reused adaptively','statistical predictors only; no neural world model',
                           'raw-router gains not preserved by deployable lattice projection',
                           '3 seeds, 1 ID capacity, 1 machine','source-domain exclusion does not guarantee concept exclusion']}
    (folder/'gate_report.json').write_text(json.dumps(result,indent=2)+'\n')
    # Report derives all numbers from the latest completed run.
    latest=sorted(runs)[-1];p=folder/latest;ms=json.loads((p/'metrics.json').read_text());by={(m['method'],m['suite']):m for m in ms}
    lines=['# 第二阶段：首批四轮探索实验','',
      '**结论：低人工介入的边界学习和快速信号推理已实现，但尚无新模型同时通过质量与速度验收。**',
      'RL 在冻结的因果字节预测代理上显示正向学习；这不是数学理解、神经语言模型或下游收益证据。','',
      '## 范围与方法','',
      '原始 benchmark 与生产模型保持冻结。仅从允许的 broad fit 池重新划分：fit 4,096（102篇）、reward 512（31篇）、dev 512（19篇）、内部测试 1,024（20篇）、数论来源域测试 512（19篇）。各 scope 按论文隔离，排除旧测试。原语料此前可见，四轮又复用这些内部集，因此全部属于自适应探索，不能当成独立确认实验。','',
      '容量 1,024；每轮同数据重训 byte BPE、数学原子 BPE、边界 AST、Unigram；原接口运行 Qwen2.5、Llama 3、o200k。所有系统使用相同未截断字符串与既有指标实现。全 public encode 接口进行7次交错计时，包括信号计算。原生 DP 的提升与信号学习贡献分别报告。','',
      '自主策略：每个两字节上下文一个 Bernoulli 边界参数；只观察过去字节；LOO reward-to-go score-function；冻结的统计 patch 预测器只用 fit 768式训练。policy reward 使用48式；三个种子17/29/43。pilot01为16轮，后面64轮，每轮4个rollout。每个64轮优化器每种子12,288条轨迹、885,248个字节目标，原 RL、随机搜索、CEM、无更新预算相同。','',
      '人工边界标注=0、命令白名单=0、AST训练标签=0、数学质量指标奖励=0。仍有架构、统计平滑、patch长度上限、预算约束、优化超参数、导出方案等人工选择；不能称为完全没有人工先验。','',
      '## 四轮结果','',
      '- pilot01：预测熵、surprise、分布变化和RL策略的导出均选零信号。高速属于C实现与最短路径，不能记成信号或RL收益。',
      '- pilot02：学习token转移概率，以整条路径预测编码打分；切分产生真实变化。RL增加到64轮。',
      '- pilot03：从信号阈值发现基础单位，再用原版HF BPE；多数指标下降，保留负结果，不晋级。',
      '- pilot04：无字符类别标注的邻域分布聚类；基础单位质量仍不够。聚类后的一字节前缀附加为明确的通用人工架构先验，不归为RL学习。','',
      '## 相同数据下的部署结果','',
      '最新轮内部论文测试；↓表示更低更好，速度与完整性↑。表中每个方法都是独立模型，不混配代理效果与编码速度。','',
      '| 方法 | token/byte↓（论文宏平均） | 编码MiB/s↑ | 命令完整性↑ | 树跨越↓ | 作用域边界召回↑ | 词法边界F1↑ |',
      '|---|---:|---:|---:|---:|---:|---:|']
    labels={'boundary_ast':'同fit边界AST','math_atom_bpe':'同fit数学原子BPE','byte_bpe':'同fit字节BPE',
            'zero_signal':'零信号原生最短路径','predictive_path':'学习预测路径','context_units_fast':'自动聚类单位＋快速路径',
            'context_units_predictive':'自动聚类单位＋预测路径'}
    for name,label in labels.items():
        m=by[(name,'internal_eval')];lines.append(f"| {label} | {m['document_macro_tokens_per_byte']:.4f} | {m['encode_mib_s']:.2f} | {100*m['command_integrity']:.2f}% | {100*m['independent_tree_crossing']:.2f}% | {100*m['independent_scope_boundary_recall']:.2f}% | {100*m['lexical_boundary_f1']:.2f}% |")
    lines+=['','数论来源域测试（未参与上述训练/选择），仍不能保证数学概念完全不重叠：','',
            '| 方法 | token/byte↓ | 编码MiB/s↑ | 命令完整性↑ | 树跨越↓ | 作用域边界召回↑ |','|---|---:|---:|---:|---:|---:|']
    for name in ('boundary_ast','zero_signal','predictive_path'):
        m=by[(name,'domain_ood')];lines.append(f"| {labels[name]} | {m['document_macro_tokens_per_byte']:.4f} | {m['encode_mib_s']:.2f} | {100*m['command_integrity']:.2f}% | {100*m['independent_tree_crossing']:.2f}% | {100*m['independent_scope_boundary_recall']:.2f}% |")
    lines+=['','## RL学习证据与限制','',
      '| 种子 | 相对随机搜索字节损失改善 | 相对CEM字节损失改善 | 相对无更新改善 |','|---|---:|---:|---:|']
    lr=runs[latest]['learning_proxy']
    for i,seed in enumerate((17,29,43)):
        lines.append(f"| {seed} | {lr['random_search']['relative_bpb_improvement_percent'][i]:.2f}% | {lr['cem_search']['relative_bpb_improvement_percent'][i]:.2f}% | {lr['no_update']['relative_bpb_improvement_percent'][i]:.2f}% |")
    lines+=['','各种子的探索性论文簇bootstrap区间方向一致；区间没有做多重比较校正，也没有完整重训总体置信区间，不能用于确认性显著性声明。策略边界率不完全相同；当前证据属于预算约束下的预测代理收益，仍需严格匹配实际rate/compute。该收益未通过固定词表投影保留，不能把这张表与前面的部署速度表合并称为一个优秀模型。','',
      '## 验收与审查','',
      '四轮分别完成至少三轮实际自动审查：数据/来源隔离；原生与Python参考、可逆性和保存加载；逐式与论文宏平均重聚合、相同oracle预算、原始payload、冻结hash及交错计时。另有独立report脚本重新检查全部结果档案与准确源码快照。所有逐式还原通过。单元测试另验证短序列全局最优、因果边界、枚举期望梯度、EOS正规化和未知输入。','',
      '按预设门槛，新系统的压缩、命令完整性、词法边界或结构跨越仍有不合格项；所有阶段二候选均未通过部署晋级，原7方法注册表没有改动。高速度或单一代理改善不足以满足用户的双重条件。','',
      '## 下一轮优先事项','',
      '1. 将RL的“随机patch策略”与部署decoder统一，避免导出过程消除学习收益；测试固定实测边界率的控制。',
      '2. 复现Dauncey公开神经架构或实现同等因果小模型，使反馈来自有计算预算控制的神经预测，而非有限状态计数器。',
      '3. 用预测反馈共同学习词表与层级单位，减少当前频率排名/阈值/聚类附加先验；保留每种先验的消融。',
      '4. 加入条件明确、符号可验证的数学操作训练数据；单独测未见操作/组合/领域，之后才研究世界模型。',
      '5. 多词表容量、更大种子集、全新来源冻结测试；只有两个成功条件同时通过才更新生产候选。','',
      '## 复现','',
      '```sh',
      '.venv/bin/python -m pytest -q',
      '.venv/bin/python scripts/run_stage2.py --output experiments/stage2/replay04 --predictive --context-units --rounds 64',
      '.venv/bin/python scripts/report_stage2.py','```','',
      '历史pilot精确源码位于各目录source_snapshot，按manifest hash匹配。所有代理概率/策略/单位/候选、每式结果和审查JSON保留。重新计时会因机器负载变化；不能要求速度数值逐位相同。']
    calibration=folder/'matched_rate_router.json'
    if calibration.exists():
        cc=json.loads(calibration.read_text());assert cc['source_sha256']==sha(ROOT/'scripts/calibrate_stage2.py')
        assert all(abs(r['target']-r['calibrated_expected_reward_rate'])<1e-5 for r in cc['results'])
        lines+=['','## 匹配边界率的额外敏感性检查','',
                '仅用48个reward公式解析计算期望边界率，对每种策略执行相同24步二分校准；不读dev损失来调偏置。dev统一8次采样。此检查为事后公平性敏感性分析，不是新确认实验。reward边界率精确一致；dev边界率仍可能因分布变化有差异。','',
                '| 种子 | RL对随机搜索改善 | RL对CEM改善 | RL对无更新改善 |','|---|---:|---:|---:|']
        for seed in (17,29,43):
            vals=[next(r['relative_bpb_improvement'] for r in cc['comparisons'] if r['seed']==seed and r['control']==control) for control in ('random_search','cem_search','no_update')]
            lines.append(f'| {seed} | {vals[0]*100:.2f}% | {vals[1]*100:.2f}% | {vals[2]*100:.2f}% |')
        lines+=['','匹配reward边界率后正向收益仍保留，但对象仍是冻结的统计预测器；尚不能声称RL整体成功或数学理解。','',
                '运行 `.venv/bin/python scripts/calibrate_stage2.py` 重做该检查；再运行report脚本更新汇报。','',
                '训练时间注意：部分投影/选择模型未登记完整逐模型训练时长（train_seconds可为0）；不得据此比较训练效率。完整实验wall time与RL训练时长已单独记录。']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Report and success gates written;',sum(r['rows'] for r in runs.values()),'result rows independently checked.')


if __name__=='__main__':main()
