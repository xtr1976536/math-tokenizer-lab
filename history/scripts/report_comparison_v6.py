"""Generate direct general-versus-mathematical tables from validated results."""
import csv
import json

from math_tokenizer.comparison_v6 import comparison_boards
from math_tokenizer.corpus import ROOT

NAMES = {'ast_v4': 'AST-v4', 'v5': 'v5（离线 RL-BPE）',
         'pathpiece_bpe_vocab': 'PathPiece', 'hf_bpe': 'HF byte BPE',
         'sentencepiece_bpe': 'SentencePiece BPE', 'sentencepiece_unigram': 'SentencePiece Unigram',
         'wordpiece_bytes': 'Byte WordPiece', 'Qwen2.5-0.5B': 'Qwen2.5',
         'Qwen/Qwen3-0.6B': 'Qwen3', 'deepseek-ai/DeepSeek-V3': 'DeepSeek-V3',
         'microsoft/Phi-4': 'Phi-4', 'mistralai/Mistral-Nemo-Base-2407': 'Mistral-Nemo',
         'tiktoken_gpt2': 'GPT-2', 'tiktoken_cl100k_base': 'cl100k', 'tiktoken_o200k_base': 'o200k',
         'meta-llama/Meta-Llama-3-8B': 'Llama 3'}


def main():
    folder = ROOT / 'results/v6/full'
    result = json.loads((folder / 'general_vs_math_rankings.json').read_text())
    audit = json.loads((folder / 'comparison_fairness_audit.json').read_text())
    replay = json.loads((folder / 'fairness_audit.json').read_text())
    assert audit['all_passed'] and replay['all_passed']
    metrics = json.loads((folder / 'metrics.json').read_text())
    mature_count=len({m['method'] for m in metrics if m['group']=='fixed_pretrained_reference'})
    total_count=7+mature_count
    formula_rows=sum(m['formulas'] for m in metrics)
    verification_path=folder/'comparison_verification.json'
    boards, combined = comparison_boards(metrics, result['weights'])
    # Refresh descriptive fields after reporting-code additions; scores must stay equal.
    assert all(combined[c]['ranking'] == result['combined'][c]['ranking'] and
               combined[c]['scores'] == result['combined'][c]['scores'] for c in combined)
    result['within_scenario'] = boards
    result['combined'] = combined
    (folder / 'general_vs_math_rankings.json').write_text(json.dumps(result, indent=2) + '\n')
    profiles = {r['key']: r for r in json.loads((folder / 'deployment_profile.json').read_text())}
    csvrows = []
    for board_id, board in boards.items():
        for name, values in board['metrics'].items():
            row = {'board': board_id, 'suite': board['suite'], 'method': name,
                   'comprehensive_rank': board['comprehensive_ranks'][name],
                   'compression_rank': board['metric_ranks']['document_macro_tokens_per_byte'][name],
                   **{k: v for k, v in values.items() if k != 'keys'}}
            for field in ('load_seconds', 'incremental_peak_mib', 'loaded_process_peak_mib'):
                row['deployment_' + field] = sum(profiles[k][field] for k in values['keys']) / len(values['keys'])
            csvrows.append(row)
    with (folder / 'general_vs_math_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(set(r) for r in csvrows))), lineterminator='\n')
        writer.writeheader(); writer.writerows(csvrows)
    lines = ['# 通用与数学专用 tokenizer：相同实验的直接比较', '',
             f'Qwen2.5、Qwen3、DeepSeek-V3、Phi-4、Mistral-Nemo、GPT-2、cl100k、o200k、Llama 3 全部参加直接比较和综合排名。每张榜同时含两个自研家族、五个同数据训练经典/论文基线和{mature_count}个成熟 tokenizer，共{total_count}种。', '',
             f'新论文测试3,465式、来源分类排除后的跨领域测试1,666式、压力测试521式全部输入原始 LaTeX 字符串；同一 KaTeX 参考树、UTF-8 字节定位、命令/数字定义、边界定义和统计程序。不删式、不截断、不添加聊天模板或 BOS/EOS。每种模型均以公开完整 encode/decode 接口计时，包含所需前端，七轮交错顺序运行。全部{len(metrics)}组结果原式精确还原率100%。', '',
             '原直接比较修订统一了跨领域的fit校准；本次又新增Llama 3，并对所有配置共同重新执行七轮交错计时。词表、已有模型、语料、参数和选择均未改变。固定 tokenizer 在专用跨领域测试中同样使用 specialized fit 计数，broad直接跨领域和两种压力场景保存各自同口径读数。校准只统计fit公式，不用测试公式拟合。此前八种成熟配置的报告和计时归档在pre_llama3_comparison/。', '',
             'Llama 3使用Meta-Llama-3-8B官方词表。Meta仓库下载需要授权；从公开NousResearch固定revision取得tokenizer.json，完整Git blob SHA-1 b197f72effb9d5ed16ee0f5663e11e4cfac2ba62 与官方仓库记录相同，字节数9,085,698，文件未修改。词表无损转换为Meta原版tokenizer.py读取的rank文件；正式榜使用Meta原版tiktoken接口。47,773条主库和521条压力公式逐ID与HF配置一致。两种接口速度均披露，但相同切分不重复计为两个算法。固定代码、来源与哈希见[data/pretrained_registry_v6.json](../data/pretrained_registry_v6.json)、[逐ID等价及HF接口读数](../results/v6/full/llama3_backend_equivalence.json)。', '',
             '这里回答“现成通用配置与本项目数学专用配置在当前公式任务上谁更好”。各 tokenizer 保留真实词表和公开训练来源：成熟 tokenizer 为50,257至200,019个ID，自研为1,024或2,048个ID。词表规模和训练历史作为条件记录；外部预训练是否覆盖这些论文无法认证。同训练语料、同ID预算的算法比较仍保存在原报告中。', '',
             '编码/解码计时不含模型加载，加载成本另列；部署内存为同导入环境下的新进程峰值增量，不等同常驻词表大小。成熟模型 train_seconds=0 表示本轮未重训，其外部训练成本未知；不能据此比较训练效率。Qwen2.5/Qwen3以及Phi-4/cl100k在这里可能有相同切分读数，质量指标并列时不得仅凭微小速度差推断稳定的质量差异。', '',
             f'综合权重沿用原先冻结值：论文等权 token/byte 30%、编码速度25%、独立树 crossing 20%、命令完整10%、数字串完整5%、fit-predictive unigram BPB10%；两个主场景各50%。每项按当前{total_count}种方法的百分位秩加权，越小越好，完整性不达标的配置排除。打分时相同读数使用平均秩，表中单指标相同读数列为并列名次；列表内部的字母排序不代表优劣。BPB是词表加半平滑的 unigram 读数，词表容量也影响它，不是模型 perplexity；压缩、结构、速度分别报告以避免综合分掩盖取舍。', '',
             '完整标量指标、长度分位数、Rényi 效率（α=1、2、2.5、3）、词表利用率、边界P/R/F1、词素保留、算子独立、UTF-8完整、布局变化、解码速度和部署资源见 [CSV](../results/v6/full/general_vs_math_metrics.csv)。公式逐式结果、领域/来源/两档复杂度/论文分组与作者原版熵读数使用同一完整结果档案。', '']
    for capacity in (1024, 2048):
        lines += [f'## 自研/重训基线 {capacity} ID 与{mature_count}种原版成熟配置', '',
                  '|综合排名|配置|综合分|新论文压缩排名|跨领域压缩排名|',
                  '|---:|---|---:|---:|---:|']
        score = combined[str(capacity)]
        for index, name in enumerate(score['ranking'], 1):
            left = boards[f'{capacity}/broad']['metric_ranks']['document_macro_tokens_per_byte'][name]
            right = boards[f'{capacity}/specialized']['metric_ranks']['document_macro_tokens_per_byte'][name]
            lines.append(f"|{score['ranks'][name]}|{NAMES[name]}|{score['scores'][name]:.4f}|{left}|{right}|")
        for scenario, label in [('broad', '新论文全量'), ('specialized', '跨领域全量')]:
            board = boards[f'{capacity}/{scenario}']
            lines += ['', f'### {label}', '',
                      '|配置|实际词表|token/式↓|paper token/byte↓|命令完整%↑|数字完整%↑|独立 crossing%↓|编码 MiB/s↑|解码 MiB/s↑|同fit BPB↓|',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for name in board['primary_ranking']:
                m = board['metrics'][name]
                lines.append(f"|{NAMES[name]}|{int(m['vocab_actual'])}|{m['tokens_per_formula']:.3f}|{m['document_macro_tokens_per_byte']:.5f}|{100*m['command_integrity']:.3f}|{100*m['number_integrity']:.3f}|{100*m['independent_tree_crossing']:.3f}|{m['encode_mib_s']:.3f}|{m['decode_mib_s']:.3f}|{m['fit_predictive_unigram_bits_per_byte']:.4f}|")
        lines += ['', '### 同一 broad 模型直接跨领域：单独核对泛化', '',
                  '这一比较使用广域训练模型在同一1,666式上测试，并保留 broad fit 校准；避免把专门训练变化同数据分布变化混为一谈。', '',
                  '|配置|token/式↓|paper token/byte↓|命令完整%↑|crossing%↓|编码 MiB/s↑|broad fit BPB↓|',
                  '|---|---:|---:|---:|---:|---:|---:|']
        board = boards[f'{capacity}/broad_ood']
        for name in board['primary_ranking']:
            m = board['metrics'][name]
            lines.append(f"|{NAMES[name]}|{m['tokens_per_formula']:.3f}|{m['document_macro_tokens_per_byte']:.5f}|{100*m['command_integrity']:.3f}|{100*m['independent_tree_crossing']:.3f}|{m['encode_mib_s']:.3f}|{m['fit_predictive_unigram_bits_per_byte']:.4f}|")
        lines += ['', '### 全量语法和长输入压力测试', '',
                  '521式为固定公式的扰动/长输入压力集，不能当作新的独立论文样本；不纳入综合主榜和论文bootstrap。', '',
                  '|配置|broad token/式|specialized token/式|broad 命令完整%|specialized 命令完整%|broad 编码 MiB/s|specialized 编码 MiB/s|',
                  '|---|---:|---:|---:|---:|---:|---:|']
        a, b = boards[f'{capacity}/broad_stress'], boards[f'{capacity}/specialized_stress']
        for name in a['primary_ranking']:
            x, y = a['metrics'][name], b['metrics'][name]
            lines.append(f"|{NAMES[name]}|{x['tokens_per_formula']:.3f}|{y['tokens_per_formula']:.3f}|{100*x['command_integrity']:.3f}|{100*y['command_integrity']:.3f}|{x['encode_mib_s']:.3f}|{y['encode_mib_s']:.3f}|")
    mature_names=[n for n,m in boards['1024/broad']['metrics'].items() if m['model_group']=='fixed_pretrained_reference']
    mature_compression={s:[NAMES[n] for n in boards[f'1024/{s}']['primary_ranking'] if n in mature_names][:2] for s in ('broad','specialized')}
    mature_overall={c:next(NAMES[n] for n in combined[str(c)]['ranking'] if n in mature_names) for c in (1024,2048)}
    top_two={c:[NAMES[n] for n in combined[str(c)]['ranking'][:2]] for c in (1024,2048)}
    speed_summary=[]
    for capacity in (1024,2048):
        values=boards[f'{capacity}/broad']['metrics']
        speed_summary.append(f"{capacity}ID：AST-v4 {values['ast_v4']['encode_mib_s']:.2f}、v5 {values['v5']['encode_mib_s']:.2f}、o200k {values['tiktoken_o200k_base']['encode_mib_s']:.2f}、Llama 3 {values['meta-llama/Meta-Llama-3-8B']['encode_mib_s']:.2f} MiB/s")
    sensitivity_summary=[{'speed_weight':x['speed_weight'],'top_two':{c:[NAMES[n] for n in b['ranking'][:2]] for c,b in x['combined'].items()}} for x in result['weight_sensitivity']]
    lines += ['', '## 直接结论与取舍', '',
              f'本轮共同重新计时后，综合前两名：{top_two}。两套主场景的压缩第一均为v5。成熟配置压缩前两名：{mature_compression}；成熟配置综合第一：{mature_overall}。', '',
              '新论文编码速度：'+'；'.join(speed_summary)+'。v5仍需提升编码速度。DeepSeek-V3和o200k的连续数字完整性高于自研模型；GPT-2的crossing低于v5，需结合它更长的token序列解释，不能称自研每一项都优于通用方法。', '',
              '速度权重敏感性（10%/20%/30%/40%，其余权重按比例缩放）：'+json.dumps(sensitivity_summary,ensure_ascii=False)+'。', '',
              'v5把命令完整性提升到新论文约99.74%，比成熟配置的2%–28%高；但这是按源字符串中命令字符是否在token内部被切开的定义，通用模型分开编码命令不代表其下游推理必然失败。结构crossing只计独立KaTeX源跨度的部分交叠，不是数学语义错误率。', '',
              '复合排名对比较对象集合也敏感，加入成熟配置或重新共同计时会使后续名次发生变化。应以逐项读数、Pareto集、计时轮次和固定比较对象一起解释，不能直接比较不同集合的百分位综合分。', '',
              f"两自研家族与{mature_count}成熟配置、两个预算、两主场景及broad直接跨领域共{len(result['paired_paper_bootstraps'])}组配对论文bootstrap（每组2,000次）已保存。95%区间以冻结词表为条件，是未作多重比较调整的逐对描述；不能当作已验证的下游优势。", '',
              '必须保留的不确定性：1024跨领域的AST-v4相对DeepSeek-V3/Mistral-Nemo，论文等权压缩差的95%区间分别约[−0.03533,0.00764]、[−0.03935,0.00990]；v5相对Mistral-Nemo约[−0.06508,0.00030]。三者均包含零，不能把点估计排名说成这些配对中已确证的压缩优势。', '',
              f'本次再做四轮专门公平性审查：全量原式/ID/字节/论文组、官方tokenizer重载与同fit校准、独立汇总和七次计时一致、全部方法进入同榜与配对统计。全模型逐式重放覆盖{formula_rows:,}行。测试通过数见[验证清单](../results/v6/full/comparison_verification.json)，记录模型、对比表、并列排名、校准和审查哈希。', '',
              '复现：先运行 `sh scripts/run_v6_replay.sh`，该脚本顺序完成原全量评测、四轮直接比较审查、全模型重放和两种排名。只复核现有冻结结果可运行 `scripts/compare_pretrained_v6.py` → `scripts/audit_full_v6.py` → `scripts/summarize_full_v6.py` → `scripts/report_comparison_v6.py`，均使用 `.venv/bin/python`。', '',
              '证据：[直接比较协议](../results/v6/full/comparison_protocol.json)、[校准修订原读数](../results/v6/full/calibration_amendment.json)、[四轮专项审查](../results/v6/full/comparison_fairness_audit.json)、[全模型重放](../results/v6/full/fairness_audit.json)、[机器排名与配对区间](../results/v6/full/general_vs_math_rankings.json)、[同预算算法结果](v6_results_and_rankings.md)。', '']
    path = ROOT / 'docs/general_vs_math_v6.md'
    path.write_text('\n'.join(lines))
    print(path, flush=True)


if __name__ == '__main__':
    main()
