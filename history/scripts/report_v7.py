"""Final results, failures, and resolved/remaining reviewer concerns."""
import json
from math_tokenizer.corpus import ROOT

NAMES={'ast_v4_parent':'AST-v4 原版','v5_parent':'v5 原版','ast_lexicographic':'AST 两级最短路径',
       'v5_terminal_transfer':'v5 终端RL迁移','official_atomic_bpe':'同原子原版BPE','hf_byte_bpe':'原版byte BPE',
       'pathpiece_atomic_vocab':'同原子词表PathPiece','pathpiece_byte_vocab':'byte词表PathPiece',
       'sentencepiece_bpe':'SentencePiece BPE','sentencepiece_unigram':'SentencePiece Unigram','wordpiece':'WordPiece',
       'Qwen/Qwen3-0.6B':'Qwen3','Qwen2.5-0.5B':'Qwen2.5','deepseek-ai/DeepSeek-V3':'DeepSeek-V3',
       'microsoft/Phi-4':'Phi-4','mistralai/Mistral-Nemo-Base-2407':'Mistral-Nemo','meta-llama/Meta-Llama-3-8B':'Llama 3',
       'tiktoken_gpt2':'GPT-2','tiktoken_cl100k_base':'cl100k','tiktoken_o200k_base':'o200k'}


def main():
    folder=ROOT/'results/v7/full'
    ranks=json.loads((folder/'rankings.json').read_text())
    metrics=json.loads((folder/'metrics.json').read_text())
    manifest=json.loads((ROOT/'data/releases/v7/manifest.json').read_text())
    audit=json.loads((folder/'fairness_audit.json').read_text())
    completion=json.loads((folder/'completion.json').read_text())
    verification=json.loads((ROOT/'results/v7/verification.json').read_text())
    selection=json.loads((ROOT/'results/v7/development_selection.json').read_text())['selection']
    assert audit['all_passed'] and verification['all_passed'] and all(not m['provisional_timing'] for m in metrics)
    boards=ranks['within_scenario']
    index_speed=json.loads((ROOT/'results/v7/training_index_speed.json').read_text())
    lines=['# v7：先验对齐、结构分词与终端RL的完整实验','',
        '本轮先修复最近邻基线与信息访问差异，再探索完整公式词表剪枝、有限结构代价、节点可信度、无权重的两级最短路径，以及只使用fit动作特征的完整轨迹RL。新AST压缩分支取得可确认的收益，但没有实现两家族所有核心指标的大幅提升；RL候选未通过开发集晋升门槛，活动模型仍为v6 AST-v4/v5。','',
        f"新数据来自{manifest['papers_downloaded']}篇固定2020年前CC BY/CC0许可核验源码，{manifest['papers_with_accepted_formulas']}篇贡献{manifest['formulas']:,}条接受公式，其中跨来源领域测试{manifest['splits']['extreme_domain_test']:,}条。排除v5/v6已暴露精确式、模板和渲染等价式，旧主库哈希未变。领域、历史来源、simple/complex两档、原稿及宏变换证据均保留。最终大小以manifest为准；前期较小快照未用于模型评分。",'',
        f"冻结{completion['configurations']}种配置，完整{completion['metric_records']}组自然/压力结果、{completion['formula_rows']:,}行逐式结果。四轮评分前加四轮评分后检查，共{audit['total_rounds']}轮通过；所有输入原文及payload无损。统一七轮交错encode/decode计时，包括前端与codec。521式压力集沿用已暴露v6输入，不称为新未见论文。",'',
        f"最终{verification['tests_passed']}项测试通过；三来源重划分的12个保存模型逐个重载，在每组256条共同输入上复算长度、树跨度、完整性和token使用计数，全部吻合。冻结时间早于第一次新测试评分、检查点哈希、活动注册表和旧主库均已核验。证据见[交付核验](../results/v7/verification.json)。",'',
        '同先验基线有共享fit派生原子＋原版HF BPE、相同原子词表的作者PathPiece，以及同词表关闭/硬/两级结构分词。经典BPE、SentencePiece BPE/Unigram、WordPiece和PathPiece均提供fit+reward联合训练，使原版算法获得相同范围资料。允许信息范围已对齐，使用方式、调优次数和耗时仍不完全相同，主榜不能排除全部算法归因混杂。','',
        'Qwen2.5/Qwen3/DeepSeek-V3/Phi-4/Mistral-Nemo/Llama3/GPT-2/cl100k/o200k参加完全相同自然、跨领域和压力实验。成熟配置保留官方词表/接口，外部预训练污染未知。各场景概率校准使用相同fit IDs与实际词表容量。','',
        '## AST新分支：收益与代价','',
        '两级DP先最少token，在所有最短路径中再最少native源树crossing，不需要惩罚系数。与BPE使用完全相同的字节词表、原子和ID预算。它与同mask仅优化长度的最短分词在每条输入上token数相同，差别来自最短路径结构择优。表中crossing仍由独立KaTeX计算，native与参考树并不相同。','',
        '|预算|场景|方法|paper token/byte↓|命令完整%↑|KaTeX crossing%↓|独立分组边界召回%↑|编码MiB/s↑|',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            values=boards[f'{capacity}/{scenario}']['metrics']
            for name in ('ast_v4_parent','v5_parent','official_atomic_bpe','ast_lexicographic','v5_terminal_transfer'):
                m=values[name]
                lines.append(f"|{capacity}|{scenario}|{NAMES[name]}|{m['document_macro_tokens_per_byte']:.5f}|{100*m['command_integrity']:.2f}|{100*m['independent_tree_crossing']:.2f}|{100*m['independent_scope_boundary_recall']:.2f}|{m['encode_mib_s']:.2f}|")
    lines+=['',
        '新的压缩分支crossing高于原版硬树AST，独立pylatexenc分组/参数边界召回低于同原子BPE。它改善长度、KaTeX crossing及本机接口速度，同时损失另一外部结构代理，不能宣称整体数学结构或所有指标占优。','',
        f"训练索引C++精确计数/更新通过逐动作、非重叠gain与源跨度等价验证。在{index_speed['formulas']}条输入、{index_speed['merges']}个合并、五轮交错对照上加速{index_speed['speedup_median']:.2f}倍；只说明该索引的工程收益，不代表完整RL训练同倍加速。",'',
        '## 终端RL、迁移与对照','',
        '24维策略只观察fit统计，在来源均衡子任务中获得实际导出BPE的终端多目标回报，执行完整轨迹RLOO更新。即时reward不进入动作特征，完整词表导出不访问reward，上线仍为固定Rust BPE。广域1024保留17/29/43三训练种子；专用1024另训严格来源策略；2048迁移同场景策略。广域策略移到专用池的实验此前接触目标学科，单列为暴露域诊断，不进入严格OOD榜。','',
        'frequency、no-update、同即时reward greedy和固定策略搜索均保留。默认RL和搜索各16次终端评分，但时间不完全相同；六轮较高tree权重RL有24次评分，另列研究分支。学习及词表导出成本分别记录，没有静默采用best sampled trajectory。','',
        '|预算/场景|RL开发集平均utility↓|父v5开发集utility↓|开发晋升|',
        '|---|---:|---:|---|']
    for label,s in selection.items():
        lines.append(f"|{label}|{s['rl_mean_dev_utility']:.6f}|{s['rl_parent_dev_utility']:.6f}|{'是' if s['rl_promoted_by_dev'] else '否'}|")
    lines+=['',
        '这些开发试验未稳定超过父v5，正式评分仍报告全部既定种子。不能将某个seed更低crossing描述成RL普遍成功。','',
        '## 冻结的主要配对统计','',
        f"共有{ranks['bootstrap_family_size']}组主配对；先对相同公式的训练seed求均值，再按论文聚类重采样2,000次。CI条件于冻结词表和观察到的seed，不代表所有随机训练的总体不确定性。负值对左侧有利，额外给出Bonferroni家族调整区间；调整尾概率很小，有限重采样尾部精度也必须考虑。",'',
        '|比较|paper token/byte差|普通95%CI|家族调整95%CI|',
        '|---|---:|---|---|']
    for name,p in ranks['primary_paired_effects'].items():
        ci=p['ci95'];adjusted=p['familywise_95_bonferroni_ci']
        lines.append(f"|{name}|{p['paper_macro_tokens_per_byte_difference']:.6f}|[{ci[0]:.6f}, {ci[1]:.6f}]|[{adjusted[0]:.6f}, {adjusted[1]:.6f}]|")
    lines+=['','## 全部成熟与自研/经典配置同榜','',
        '综合分沿用冻结权重：压缩30%、编码25%、KaTeX crossing20%、命令10%、数字5%、fit BPB10%，两个主场景各50%。当前方法百分位加权，相同读数平均秩。它是特定偏好下的部署排序，不是科学贡献分。新独立探针没有事后加入综合分。']
    for capacity,combined in ranks['combined'].items():
        lines+=['',f'### {capacity} ID自研与原版成熟词表','',
                '|综合位置|方法|综合分↓|广域压缩名次|专用跨域压缩名次|',
                '|---:|---|---:|---:|---:|']
        for i,name in enumerate(combined['ranking'],1):
            a=boards[f'{capacity}/broad']['metric_ranks']['document_macro_tokens_per_byte'][name]
            b=boards[f'{capacity}/specialized']['metric_ranks']['document_macro_tokens_per_byte'][name]
            lines.append(f"|{i}|{NAMES.get(name,name)}|{combined['scores'][name]:.4f}|{a}|{b}|")
    lines+=['',
        '所有主指标见[CSV](../results/v7/full/all_main_metrics.csv)，单场景、Pareto、seed范围和权重敏感性见[机器排名](../results/v7/full/rankings.json)。所有控制/研究分支也在metrics.json和逐式archive中。','',
        '## 独立验证、泛化与未解决问题','',
        'pylatexenc分组/参数端点边界未用于训练奖励；局部同字节符号编辑测16字节邻域外的边界变化；概率读数用α=0.01/0.1/0.5/1平滑。均为共同输入和探针选择，并报告解析覆盖。这些是源码本征诊断，不是普遍认可的数学语义金标准。字节分词边界召回高、整式token可低crossing的退化反例测试保留。','',
        '三轮来源论文重划分，每轮四方法从头学习词表/策略：同词表AST分词减少长度并降低KaTeX crossing，终端RL没有稳定超过no-update。这些重复实验使用早已可访问的v6 train/dev来源，不称为新增封存测试。按领域/来源/复杂度/论文的全量汇总见grouped_metrics.json。','',
        '144条源码/结构双审材料已准备，reviewer1/2答案为空，专家审核完成数为0。当前仅一台Mac上的公共接口七轮计时，没有第二硬件或独立团队复现。','',
        '|严格评审问题|本轮处理|剩余证据|',
        '|---|---|---|',
        '|最近邻基线与信息访问|同原子BPE/PathPiece、fit+reward经典训练、同词表结构开关|近期词表学习法和计算/调优预算仍需补齐|',
        '|AST归因|相同词表的关闭/硬/有限/两级路径；小图穷举及整库长度核验|独立参数边界召回下降，数学语义收益未建立|',
        '|RL必要性|终端轨迹、三seed、迁移、no-update/frequency/greedy/16评分搜索|额外学习收益未稳定，未晋升|',
        '|指标有效性|pylatexenc、局部编辑、平滑敏感性和退化反例|仍是语法代理，专家gold未完成|',
        '|泛化统计|31篇有效新来源、旧模板排除、三来源resplit、聚类及家族调整区间|新库领域覆盖不均，新数论论文不足；概念未证明完全不重合|',
        '|理论与原创性|固定合法图、两级DP精确性、mask保护条件，直接列出先例|组合贡献仍需定位，巨大突破主张不成立|',
        '|工程速度|C/C++核、等价验证、公共接口七轮交错计时|单机、非批量/非多线程范围|','',
        '下一轮应围绕同时保留参数边界和压缩的结构机制，以及稳定超过同reward简单控制的RL credit继续研究。v7自然测试现在也已暴露，未来配置只能在dev选择并另封存最终验证。没有达到的“大幅提升”不会描述成完成。','',
        '顺序重放：sh scripts/run_v7_replay.sh。机制与先例见[v7方法](v7_formal_mechanisms.md)，实验边界见[研究协议](v7_research_protocol.md)，依赖见requirements-v7.lock.txt。完整证据在results/v7/full；待审材料在data/releases/v7/review_materials。']
    path=ROOT/'docs/v7_results.md';path.write_text('\n'.join(lines)+'\n');print(path,flush=True)


if __name__=='__main__':main()
