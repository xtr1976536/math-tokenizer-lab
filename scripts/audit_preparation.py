"""Independent saved-output checks, true encoding timing, and Chinese report."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
import statistics
import time
from math_tokenizer.corpus import ROOT
from math_tokenizer.pde_protocol import split_corpus,digest
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet,train_compact
from scripts.当前四分词器 import load


def main():
    out=ROOT/'experiments/upstream_811'
    rows=list(map(json.loads,gzip.open(ROOT/'data/formulas/pde/pde_equations.jsonl.gz','rt')))
    plans,split=split_corpus(rows)
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['split']==split
    assert protocol['data_sha256']==digest((ROOT/'data/formulas/pde/pde_equations.jsonl.gz').read_bytes())
    results=json.loads((out/'validation_results.json').read_text())
    evaluated=[r for r in results if r['status']=='evaluated']
    references={}
    for row in evaluated:
        name,size=row['alphabet'],row['total_ids'];folder=out/name
        model=CompactAlphabet.load(folder/f'{size}.model.json')
        assert len(model.vocab)+3==size
        assert digest(json.dumps([x.hex() for x in model.vocab]))==row['vocabulary_sha256']
        for part in ('train','validation'):
            saved=list(map(json.loads,gzip.open(folder/f'{size}.{part}.jsonl.gz','rt')))
            assert [r['id'] for r in saved]==[r['id'] for r in plans[part]]
            totals={k:sum(r[k] for r in saved) for k in ('initial_character_tokens','final_tokens','commands','commands_intact','numbers','numbers_intact')}
            for metric,num,den,scale in [('tokenizer compression','initial_character_tokens','final_tokens',1),('operation','commands_intact','commands',100),('numbers','numbers_intact','numbers',100)]:
                assert abs(totals[num]/totals[den]*scale-row[part][metric])<1e-10
            signature=(totals['initial_character_tokens'],totals['commands'],totals['numbers'])
            assert references.setdefault(part,signature)==signature
        # Full validation replay from frozen files, not in-memory training object.
        for formula in plans['validation']:
            assert model.decode(model.encode(formula['latex']))==formula['latex']
    # A real-corpus independent retraining checks prefix equivalence beyond toy tests.
    config=protocol['alphabets']['latex_ascii'];alpha=config['alphabet']
    model=train_compact('ast_bpe',[r['latex'] for r in plans['train']],128,alpha)
    frozen=CompactAlphabet.load(out/'latex_ascii/128.model.json')
    assert model.vocab==frozen.vocab
    assert [tuple(m) for m in model.internal.merges]==[tuple(m) for m in frozen.internal.merges]
    for row in rows:
        assert model.encode(row['latex'])==frozen.encode(row['latex'])
    selection=json.loads((out/'selection.json').read_text())
    test=json.loads((out/'test_results.json').read_text())
    for row in test:
        assert row['test_ids_sha256']==digest('\n'.join(r['id'] for r in plans['test']))
        assert row['full_coverage']
    # Pure encode timing excludes lexeme metrics and decoding. Same validation input.
    texts=[r['latex'] for r in plans['validation']]
    timing=[]
    arms=[(r['alphabet']+'/'+str(r['total_ids']),CompactAlphabet.load(out/r['alphabet']/f"{r['total_ids']}.model.json"))
          for r in evaluated if r['total_ids'] in (128,256,512)]
    arms += [(name,load(name)) for name in ('qwen2.5','llama3','o200k_base')]
    for name,m in arms:
        for text in texts[:5]:m.encode(text)
        times=[]
        for _ in range(5):
            start=time.perf_counter();tokens=sum(len(m.encode(t)) for t in texts);times.append(time.perf_counter()-start)
        elapsed=statistics.median(times)
        timing.append(dict(model=name,formulas=len(texts),tokens=tokens,median_encode_seconds=elapsed,
                           input_bytes_per_second=sum(len(t.encode()) for t in texts)/elapsed,repeats=5))
    (out/'encoding_timing.json').write_text(json.dumps(timing,indent=2)+'\n')
    prep=ROOT/'data/formulas/downstream_preparation'
    audit=json.loads((prep/'audit.json').read_text())
    records=list(map(json.loads,(prep/'conversion_records.jsonl').read_text().splitlines()))
    assert len(records)==len(rows) and {r['id'] for r in records}=={r['id'] for r in rows}
    assert all(r['roundtrip_verified'] and r['support']==r['reconstructed_support'] and not r['matched_targets'] for r in records if r['status']=='common_eligible')
    counts=Counter(r['status'] for r in records)
    assert counts==audit['status_counts']
    # Explicitly separate uncertified inputs from mathematically impossible ones.
    with (prep/'needs_review.jsonl').open('w') as f:
        for r in records:
            if r['status']=='rejected':f.write(json.dumps(r,ensure_ascii=False)+'\n')
    checks=dict(status='PASS',matrix_rows=len(results),trained_validation_arms=len(evaluated),
                full_validation_roundtrip=True,persisted_counts_reaggregated=True,metric_denominators_identical=True,
                independent_128_training_equals_saved_prefix_on_all_1500=True,
                source_formula_content_unchanged=True,common_corpus_checks=True,
                test_selection_file_sha256=digest((out/'selection.json').read_bytes()),
                timing='five repeats; median; encode only; same 150 validation formulas')
    (out/'independent_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    report=ROOT/'reports/preparation_811';report.mkdir(parents=True,exist_ok=True)
    lines=['# PDE 语料准备与 8:1:1 上游实验','',
           '本轮使用完整 1500 条原始 PDE LaTeX。上游不进行 GRE 筛选或八案例排除；下游只准备语料，尚未启动新的 Transformer 训练及系数拟合。','',
           '## 数据划分','',
           f"训练 / 验证 / 测试：{split['counts']['train']} / {split['counts']['validation']} / {split['counts']['test']}。将来源、模板哈希及规范化哈希的连接关系合并后划分，共 {split['connected_groups']} 组，最大组 {split['largest_connected_group']} 条。1500 个 ID 全部保留。",'',
           '划分文件和所有基础表、容量、选择规则见 `experiments/upstream_811/protocol.json`。现有结构哈希不能证明任意数学等价式均已隔离；这一限制不能写成全面结构等价隔离。','',
           '## 基础字符与总词表组合','',
           '总词表大小包括基础字节、学习得到的合并 token 和 3 个控制符；无隐藏 fallback。命令和变量组合不预留为基础 token。原始文本不改写。','',
           '| 基础表 | 基础字符数 | 训练集覆盖 | 说明 |','|---|---:|---|---|']
    for name,conf in protocol['alphabets'].items():
        missing=set().union(*(set(r['latex'].encode()) for r in plans['train']))-set(conf['alphabet'])
        lines.append(f"| {name} | {len(conf['alphabet'])} | {'完整' if not missing else '不足：'+', '.join(repr(chr(c)) for c in sorted(missing))} | {conf['rule']} |")
    lines += ['',f"共 {len(results)} 个组合，其中 {len(evaluated)} 个完成训练与验证；{sum(r['status']=='below_base_plus_controls' for r in results)} 个总容量小于基础表加控制符，{sum(r['status']=='training_characters_not_covered' for r in results)} 个训练字符覆盖不足。",'',
              '原始 1500 条文本共有 90 种字符；训练观测表另保留未出现的数字 9，因此为 91 个基础字节。固定 latex_ascii 表同样是 91 字符，且与本次训练观测表恰好一致。后者是依赖训练集的诊断方案，不能当作另一种独立固定字符协议。','',
              '直接字符表示下，不能把 64-ID 方案作为全语料的可运行对照。若将来研究转义或其他编码，需另立协议、计入所有 ID，并验证 AST 边界；本轮没有以这些变换改变原始算法。','',
              '## 验证集结果','',
              '三指标沿用既有公式定义。compression 是总源字节数 / 总输出 token 数；operation 与 numbers 是内部没有切分边界的相应词法单元比例（%）。','',
              '| 基础表 | 总 ID | tokenizer compression | operation (%) | numbers (%) |','|---|---:|---:|---:|---:|']
    for r in evaluated:
        if r['total_ids'] in (96,128,256,512):
            v=r['validation'];lines.append(f"| {r['alphabet']} | {r['total_ids']} | {v['tokenizer compression']:.4f} | {v['operation']:.4f} | {v['numbers']:.4f} |")
    old=next(r for r in evaluated if r['alphabet']=='ascii98' and r['total_ids']==256)
    new=next(r for r in evaluated if r['alphabet']=='latex_ascii' and r['total_ids']==256)
    improvement=(new['validation']['tokenizer compression']/old['validation']['tokenizer compression']-1)*100
    lines += ['',f"在相同 256-ID 预算下，基础表从 98 减至 91，验证压缩比相对提高 {improvement:.2f}%，operation 从 {old['validation']['operation']:.4f}% 提高至 {new['validation']['operation']:.4f}%，numbers 均为 {new['validation']['numbers']:.4f}%。这支持去掉未使用基础字符，但不支持把全部原文的基础表缩至 44 或 64 字符。",'',
              '压缩比随可用合并预算增加而提高；本轮验证 Pareto 前沿选中 91 字符 / 512-ID，保留 98 字符 / 256-ID 为预先定义的历史配置对照。不能将这两个不同总容量的测试结果直接归因为基础字符表差异。没有据此确定下游最优容量。','',
              '## 冻结选择后的测试结果','',
              '完整容量矩阵保存在验证集；测试仅运行冻结的 Pareto 方案、预先定义的历史配置和三个通用分词器。未在测试集上扫描全部组合。','',
              '| 方法 | 基础表 | 总 ID | tokenizer compression | operation (%) | numbers (%) |','|---|---|---:|---:|---:|---:|']
    for r in test:lines.append(f"| {r['method']} | {r.get('alphabet','官方词表')} | {r['total_ids']} | {r['tokenizer compression']:.4f} | {r['operation']:.4f} | {r['numbers']:.4f} |")
    lines += ['', 'operation 按既有定义包含反斜杠与完整命令名。通用分词器若将反斜杠单独切开，即计为未完整保留；此数值不表示不能识别运算的语义。','',
              '## 下游 GRE 转换与目标排除','',
              f"全量逐条处理 1500 条，目前自动认证 {audit['converted_before_target_exclusion']} 条，排除八案例同结构命中 {audit['target_excluded']} 条，剩余共同候选 {audit['common_eligible']} 条。",'',
              '**这只是当前保守转换器的认证下界，不能解释为其余公式在数学上都无法进入 GRE。** 方程组、来源专用记号、未确定维度的算子等需要进一步组件与上下文审查。尚未完成这些人工/语义复核，因此下游可用语料规模尚未最终确定。','',
              '认证包含符号展开与重建、已有 GRE/LaTeX 编解码往返、原作者数值项实现与独立符号导数的对照，以及 Hard 对原文和规范化文本的可逆编码。任意未知函数、复数系数、不明确的微分对象不会被当作常数删除。未知标量系数按常数处理的假设逐条披露。','',
              '| 未获认证的首个原因 | 数量 |','|---|---:|']
    for reason,n in sorted(audit['rejection_reasons'].items(),key=lambda x:-x[1]):lines.append(f'| {reason} | {n} |')
    lines += ['', '目标排除采用去系数、精确代数/乘积求导展开后的项支持，并考虑标量字段改名和空间坐标置换。没有声称覆盖任意坐标变换、积分变换或全部数学等价关系。','',
              '| 排除案例 | 数量 |','|---|---:|']
    for case,n in audit['excluded_by_case'].items():lines.append(f'| {case} | {n} |')
    lines += ['', '当前共同候选数量很小，不能直接用它替代原 221 条语料开展正式扩大语料实验。后续应优先复核未认证记录，区分解析器覆盖不足、可通过来源上下文消除的歧义与固定 GRE 表示的真实限制。','',
              '## 文件与复现','',
              '- `scripts/run_upstream_811.py`：完整上游实验；支持依据同一协议继续运行。',
              '- `scripts/prepare_downstream_corpus.py`：全量转换、数值项审计及目标排除。',
              '- `scripts/audit_preparation.py`：独立重算指标、真实语料独立训练对照、编码计时及生成本报告。',
              '- `experiments/upstream_811/validation_summary.csv`：全部 132 个组合及有效性状态。',
              '- `experiments/upstream_811/encoding_timing.json`：相同验证输入、5 次纯编码计时的中位数，排除指标计算时间。',
              '- `data/formulas/downstream_preparation/conversion_summary.csv`：逐公式状态、原文和规范化表示。',
              '- `data/formulas/downstream_preparation/needs_review.jsonl`：尚未获认证的全部记录。',
              '- `data/formulas/downstream_preparation/common_eligible.jsonl`：当前通过检查且排除目标后的共同候选。',
              '- `data/formulas/downstream_preparation/target_excluded.jsonl`：目标命中和排除证据。','',
              '运行环境使用相邻 `math_tokenizer_project/.venv`。仓库中旧实验、原 EqGPT 文件及原始数据均未改动；本轮只新增代码和实验材料。未发布到远端。','']
    (report/'实验结果与限制.md').write_text('\n'.join(lines))
    (report/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    shutil.copyfile(out/'validation_summary.csv',report/'validation_summary.csv')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
