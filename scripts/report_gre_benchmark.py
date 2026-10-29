"""Verify saved benchmark and per-row review, publish readable Chinese records."""
from collections import Counter
import csv,gzip,json
from pathlib import Path
import shutil
from math_tokenizer.corpus import ROOT
from math_tokenizer.pde_protocol import digest,split_corpus
from math_tokenizer.eqgpt.corpus_conversion import convert,gre_expression
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet,train_compact
from scripts.build_gre_external import reference221
from scripts.prepare_downstream_corpus import invariant_support,words_to_ids,TARGETS


def main():
    base=ROOT/'data/formulas';external=base/'gre_external';review=base/'pde1500_gre_review'
    out=ROOT/'reports/gre_benchmark';out.mkdir(exist_ok=True)
    manifest=json.loads((external/'manifest.json').read_text())
    audit=json.loads((review/'audit.json').read_text())
    records=list(map(json.loads,(external/'benchmark.jsonl').read_text().splitlines()))
    original=list(map(json.loads,gzip.open(base/'pde/pde_equations.jsonl.gz','rt')))
    rowreview=list(map(json.loads,(review/'row_review.jsonl').read_text().splitlines()))
    comps=list(map(json.loads,(review/'components.jsonl').read_text().splitlines()))
    assert len(rowreview)==1500 and {r['id'] for r in original}=={r['id'] for r in rowreview}
    assert len(comps)==sum(r['component_count'] for r in rowreview)
    old=reference221();targets={invariant_support(gre_expression(words_to_ids(w))) for w in TARGETS.values()}
    seen=set()
    for r in records:
        c=convert(r['latex']);assert c['gre_ids']==r['conversion']['gre_ids']
        key=invariant_support(gre_expression(c['gre_ids']))
        assert key not in old and key not in seen and key not in targets;seen.add(key)
        assert not r['original1500_source_overlap'] and not r['original1500_known_structure_overlap']
        assert all(0<=i<57 for i in c['gre_ids']) and c['support']==c['reconstructed_support']
        assert digest((ROOT/r['source_file']).read_bytes())==r['source_sha256']
        assert r['fits_current_50_positions']==(r['gre_length']<=50)
    for name,value in manifest['files_sha256'].items():assert digest((external/name).read_bytes())==value
    chars=sorted(set().union(*(set(r['latex'].encode()) for r in original)))
    model=CompactAlphabet.load(ROOT/'experiments/upstream_811/latex_ascii/512.model.json')
    for r in original:assert model.decode(model.encode(r['latex']))==r['latex']
    plans,_=split_corpus(original)
    exact90=train_compact('ast_bpe',[r['latex'] for r in plans['train']],128,chars)
    for row in original:
        assert exact90.decode(exact90.encode(row['latex']))==row['latex']
    counterexamples=[]
    for text in [r'\text{what?}',r'\#', 'u_t+αu_x=0']:
        try:model.encode(text)
        except ValueError as exc:counterexamples.append(dict(latex=text,error=str(exc)))
        else:raise AssertionError('Expected 91-byte coverage failure')
    oov=dict(corpus_unique_bytes=len(chars),corpus_characters=''.join(map(chr,chars)),
             declared_base_bytes=len(model.alphabet),reserved_but_absent_bytes=sorted(model.alphabet-set(chars)),
             current_corpus_minimum_single_byte_content_ids=len(chars),controls=3,
             minimum_current_corpus_total_ids_without_full_digit_reserve=len(chars)+3,
             independently_trained_90_base_128_total_model_roundtrip_all1500=True,
             current_protocol_initial_ids=len(model.alphabet)+3,
             arbitrary_UTF8_lossless_fallback_base_bytes=256,universal_91_claim=False,counterexamples=counterexamples)
    (out/'oov_scope.json').write_text(json.dumps(oov,ensure_ascii=False,indent=2)+'\n')
    lines=['# GRE 扩展 benchmark 与原 1500 条复核','','## 本次交付','',
       f"新增 benchmark 当前包含 **{len(records)} 个去重、无 GRE OOV 的结构**，来源于 {manifest['source_groups']} 个来源组。其中 {manifest['explicit_source_equations']} 条是原文显式方程，{manifest['declared_family_or_dimension_instances']} 条是原文参数族或维度分支的明确实例。它们不等于同样数量的独立论文或物理家族。",'',
       f"{manifest['fits_current_50_positions']} 条还满足现有 50-token 位置长度限制。其余公式词表可表示，但长度不能直接沿用当前模型设置。",'',
       '全部与原 EqGPT 221 行逐一比较。去系数、精确展开与空间坐标置换后，原表对应 192 个支持结构；新 benchmark 与它们不重合。仍不声称排除了任意变量变换等全部数学等价。', '',
       '现有 1500 条之外的含义：新来源不在原来源组中，且与原 1500 条中已成功转换的所有分量结构比较无重合。对原集合尚未解析的结构，无法保证不存在未识别的数学等价形式。','',
       '## 采集与筛选','',
       f"共保留 {manifest['candidate_records']} 条候选及其来源证据。筛选结果：`{json.dumps(manifest['status_counts'],ensure_ascii=False)}`。没有把换系数、换项顺序当作新结构。",'',
       '| 家族 | 去重结构数 |','|---|---:|']
    for name,n in manifest['families'].items():lines.append(f'| {name} | {n} |')
    lines+=['','参数族范围在每条 parameters 字段中公开，扩展时应按来源或完整家族分组，不能将近似实例随机分散到训练和测试。此批是有来源的初版 benchmark，尚不是上千独立 PDE 结构。','',
        '## 来源','',
        '- arXiv 原文 HTML 和 MathML alttext 已保存，包含公式编号、原式及哈希。',
        '- Journal of Differential Equations 的 Swift–Hohenberg 来源：Elizabeth Makrides、Björn Sandstede，2019，266，1073–1120，DOI 10.1016/j.jde.2018.07.064。期刊页面的 (1.1)、(1.2) 显式允许一维和二维分支，本次选一维并精确展开算子。NSF 开放库论文和 Crossref 元数据已保存。',
        '- 数据库检索包括 EqWorld/EqArchive；端点出现 TLS/超时错误，因此未将无法读取原式的数据库条目加入 benchmark。检索发现不等于已完成公式核验。','',
        '## 原 1500 条逐条复核','',
        f"全部 1500 个原始 ID 均有复核结果，布局展开后共 {audit['components']} 个分量。结果：`{json.dumps(audit['row_status_counts'],ensure_ascii=False)}`。",'',
        f"其中 {audit['whole_expression_eligible']} 条原始记录整条通过；另一些记录只获得独立分量。合计 {audit['component_status_counts'].get('eligible_component',0)} 个非目标候选分量，以及 {audit['component_status_counts'].get('target_excluded',0)} 个八案例目标命中分量。每条都保留 parent_id，不能将一个系统的某个分量写成整个系统已转换。",'',
        '这一轮增加了方程组分量拆解、域条件分离、带依据的 Laplacian 维度展开，以及单一标量场记号规范化。同时修复耦合场被误当常数、微分对象内未知系数被静默求导消去的风险；新增测试覆盖这些问题。来源说明系数为函数的表达不认证为常数系数方程。','',
        '**逐条复核指每个原始记录与分量均运行规则检查并保存结论，不能等同于对 1500 条原文逐条完成了人工数学证明。** 未通过仍分为语法/记号未覆盖、上下文不足、耦合系统、固定词表限制等，不标成全部“不可能转换”。','',
        '| 尚未获认证的首个原因 | 分量数 |','|---|---:|']
    for name,n in sorted(audit['unresolved_reasons'].items(),key=lambda x:-x[1]):lines.append(f'| {name} | {n} |')
    lines+=['','## 基础字符表与 OOV','',
       '91 不是通用下限。当前 1500 条原文有 90 种单字节字符；补齐没有出现的数字 9 后，才得到 91 个基础字符。在严格的一字符一基础 ID、原文不变、无 fallback 的表示下，覆盖当前封闭语料需要 90 个内容 ID，另加控制符；当前协议保留完整数字集，因此是 91+3=94 个初始 ID。另独立训练了一个 90 字符基础表、128 总 ID 的模型，1500 条原文往返全部通过，直接验证了 91 不是当前封闭语料的严格下限。','',
       '91 个字符不能保证未来任意 LaTeX 无 OOV：问号、井号、直接输入的希腊字母等均有实测反例。任意 UTF-8 文本的标准字节保底需要全部 256 字节（另计控制符）；转义编码等办法可以用更小底座，但改变表示和 AST 边界处理，应独立研究。','',
       '## 文件','',
       '- `data/formulas/gre_external/benchmark.jsonl`：全部已通过的新增结构及转换证据。',
       '- `data/formulas/gre_external/benchmark_maxpos50.jsonl`：当前 50-token 可用子集。',
       '- `data/formulas/gre_external/candidate_records.jsonl`：包含所有排除和重复记录。',
       '- `data/formulas/pde1500_gre_review/row_review.csv`：1500 条逐条结论。',
       '- `data/formulas/pde1500_gre_review/components.jsonl`：所有分量、转换输入、条件、结果或失败原因。',
       '- `reports/gre_benchmark/新增公式清单.md`：可读的原式、GRE 与来源清单。','']
    (out/'结果与范围.md').write_text('\n'.join(lines))
    readable=['# 新增 GRE 公式清单','', '编号仅为本次清单顺序；完整 ID、来源原式和参数实例信息见 JSONL。','']
    for i,r in enumerate(records,1):
        readable += [f"## {i}. {r['family']}",'',f"ID：`{r['id']}`；类别：`{r['origin']}`；参数：`{r['parameters']}`。",'',
           f"来源：[原文 {r['source_location']}]({r['source_url']})",'', '转换输入：','','```latex',r['latex'],'```','',
           'GRE：','','```text',', '.join(r['conversion']['gre_words']),'```','',
           f"长度：{r['gre_length']}；现有 50-token 上限：{'满足' if r['fits_current_50_positions'] else '超出'}。",'']
    (out/'新增公式清单.md').write_text('\n'.join(readable))
    accepted=[r for r in comps if r['status'] in ('eligible_component','target_excluded')]
    readable=['# 原 1500 条中本轮通过的分量','', '原始记录、原分量和转换后输入分别保存，目标命中不进入共同语料。','']
    for r in accepted:
        readable += [f"## {r['parent_id']} / 分量 {r['component_index']}",'',r['status'],'',f"[来源]({r['source_url']})",'',
           '```latex',r['component_latex'],'```','','转换输入：','','```latex',r['conversion_input'],'```','',
           '```text',', '.join(r['gre_words']),'```','',f"规则：{r['normalization_rules']}；维度证据：{r['dimension_evidence']}；目标命中：{r.get('matched_targets',[])}。",'']
    (out/'原1500条可转换分量.md').write_text('\n'.join(readable))
    checks=dict(status='PASS',all_1500_ids_reviewed=True,all_components_accounted=True,
        new_unique_structures=len(records),all_221_reference_rows_compared=True,
        new_formula_symbolic_and_codec_replay=True,all_source_hashes_verified=True,
        no_GRE_OOV=True,position_limit_separately_reported=True,source_family_instances_explicit=True,
        original1500_full_Hard_roundtrip=True,arbitrary91_no_OOV_claim_disproved=True)
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    logs=out/'logs';logs.mkdir(exist_ok=True)
    for source,name in [('/tmp/gre1500_review.log','review1500.log'),('/tmp/build_gre_external.log','external_build.log'),('/tmp/gre_review_tests.log','tests.log')]:shutil.copyfile(source,logs/name)
    for folder,description in [(external,'新增外部 GRE benchmark'),(review,'原 1500 条 GRE 复核')]:
        (folder/'README.md').write_text('# '+description+'\n\n完整说明见 `reports/gre_benchmark/结果与范围.md`。保留全部原始 ID 和来源证据，不覆盖原上游语料。\n')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
