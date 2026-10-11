"""Human-readable results generated solely from the validated final artifacts."""
import json
from pathlib import Path
from math_tokenizer.corpus import ROOT

NAMES={'ast_v4':'AST-v4（原生混合 AST/CST）','v5':'v5（结构奖励 RL）','hf_bpe':'HF byte BPE','sentencepiece_bpe':'SentencePiece BPE','sentencepiece_unigram':'SentencePiece Unigram','wordpiece_bytes':'字节 WordPiece','pathpiece_bpe_vocab':'PathPiece（作者原版）'}


def main():
    folder=ROOT/'results/v6/full';metrics=json.loads((folder/'metrics.json').read_text());ranks=json.loads((folder/'rankings.json').read_text());manifest=json.loads((ROOT/'data/releases/v6/manifest.json').read_text());audit=json.loads((folder/'fairness_audit.json').read_text())
    lines=['# v6 完整本征实验与排名','',f"语料 {manifest['formulas']:,} 条，新增 {manifest['new_paper_formulas']:,} 条来自本轮新增论文；共取得 {manifest['papers_downloaded']} 篇许可核验源码，其中本轮新增 {manifest['new_papers']} 篇。主库按领域、来源和两档复杂度分类，来源变量不混用。",
        '',f"新论文测试 {manifest['splits']['fresh_test']:,} 条；严格来源分类排除后的极端领域测试 {manifest['splits']['extreme_domain_test']:,} 条。极端训练/开发仅使用 arXiv 可核验来源，排除混合库及目标交叉主题。所有模型共用完整测试输入，不删式、不截断。公平性审查 {audit['total_fairness_rounds']} 轮通过。",
        '', '指标和权重见 [metrics_v6.md](metrics_v6.md)，方法与复现见 [v6_design_and_reproduction.md](v6_design_and_reproduction.md)。压缩用按论文等权 token/UTF-8 byte 排名；完整性、独立 KaTeX source-span crossing、概率读数和速度各有具体定义，不把熵当作下游质量保证。','']
    for capacity in (1024,2048):
        lines+= [f'## 相同数据与 {capacity} ID 预算','', '综合场景各 50%，指标权重：压缩30%、编码速度25%、独立结构20%、命令10%、数字串5%、fit-predictive BPB10%。分数是方法百分位加权，越小越好。','', '|综合排名|方法|综合分|新论文压缩排名|极端领域压缩排名|','|---:|---|---:|---:|---:|']
        board=ranks['combined'][str(capacity)]
        for i,name in enumerate(board['ranking'],1):
            left=ranks['within_scenario'][f'{capacity}/broad']['primary_ranking'].index(name)+1;right=ranks['within_scenario'][f'{capacity}/specialized']['primary_ranking'].index(name)+1
            lines.append(f"|{i}|{NAMES.get(name,name)}|{board['scores'][name]:.4f}|{left}|{right}|")
        for scenario,suite,label in [('broad','fresh_test','新论文'),('specialized','extreme_domain_test','极端跨领域')]:
            lines+=['',f'### {label} 全量结果','', '|方法|token/公式|paper token/byte|命令完整%|数字串完整%|独立树 crossing%|编码 MiB/s|predictive BPB|','|---|---:|---:|---:|---:|---:|---:|---:|']
            ms=[m for m in metrics if m['capacity']==capacity and m['scenario']==scenario and m['suite']==suite and m['group']=='matched_primary']
            for m in sorted(ms,key=lambda m:m['document_macro_tokens_per_byte']):
                name='v5' if m['method'] in ('v5_structural','v5_trajectory') else m['method']
                lines.append(f"|{NAMES.get(name,name)}|{m['tokens_per_formula']:.3f}|{m['document_macro_tokens_per_byte']:.5f}|{100*m['command_integrity']:.3f}|{100*m['number_integrity']:.3f}|{100*m['independent_tree_crossing']:.3f}|{m['encode_mib_s']:.3f}|{m['fit_predictive_unigram_bits_per_byte']:.4f}|")
        lines+=['', 'Pareto 集：'+', '.join(NAMES.get(n,n) for n in ranks['within_scenario'][f'{capacity}/broad']['pareto'])+'。单一综合权重不具有普适最优性；完整权重敏感性保存在 rankings.json。','']
    lines+=['## 固定成熟大词表参考','', '这些 tokenizer 的词表和预训练来源不同，只用于现成配置选择，不参与同训练预算的算法因果排名。官方配置未修改；是否见过相同论文无法认证。','', '|配置|实际词表|token/公式|命令完整%|树 crossing%|编码 MiB/s|原文还原%|','|---|---:|---:|---:|---:|---:|---:|']
    for m in sorted([m for m in metrics if m['scenario']=='pretrained' and m['suite']=='fresh_test'],key=lambda m:m['tokens_per_formula']):
        lines.append(f"|{m['method']}|{m['vocab_actual']}|{m['tokens_per_formula']:.3f}|{100*m['command_integrity']:.3f}|{100*m['independent_tree_crossing']:.3f}|{m['encode_mib_s']:.3f}|{100*m['roundtrip_rate']:.3f}|")
    lines+=['','## 增益归因和限制','', '词法原子属于显式结构先验，原生接口属于工程优化；它们的收益不能归入 RL 学习效果。v5 的频率、reward-greedy、关闭更新和等预算搜索控制全部保留，逐式结果与 source-cluster 95% CI 可复算。确定性期望梯度只算一个独立结果，重复 seed 不作为额外统计证据。',
        '', '屏障分支若选择空屏障，只证明策略退回频率 BPE；其较短训练不能描述为新的分词质量提升。完整轨迹 RL 和局部期望梯度不是用户上传论文的 LM/gater 联合训练复现。奖励含编码、词法、结构和稳定性，仍不是完全免除设计者选择。',
        '', 'AST 前端保留算术子集表达式 AST 与明确的语法回退。独立 KaTeX 参考是 source-span syntax AST，不是数学语义金标准；crossing 指部分区间交叠，不是任意跨多个完整叶节点。数字指标测连续数字字符串，不能当作语义数字识别。',
        '', '原始数据宏分组错误与迁移来源排除错误已修正；受影响模型和部分评分作废。修正过程不查看测试成绩选择配置，迁移重训继承测试前固定参数。当前结果不证明下游任务优越性，下一阶段需使用冻结 tokenizer 做下游实验。',
        '', '完整机器结果：results/v6/full/metrics.json、per_formula.jsonl.gz、grouped_metrics.json、rankings.json、fairness_audit.json；训练阶段报告、参数和源码审查另存 results/v6 和 data/releases/v6。']
    path=ROOT/'docs/v6_results_and_rankings.md';path.write_text('\n'.join(lines)+'\n');print(path,flush=True)


if __name__=='__main__':main()
