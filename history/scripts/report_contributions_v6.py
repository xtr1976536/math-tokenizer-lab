"""Contributions supported by implementation equivalence and RL controls."""
from collections import defaultdict
import gzip
import json

from math_tokenizer.corpus import ROOT
from summarize_full_v6 import paired_bootstrap


def main():
    folder=ROOT/'results/v6/full'
    metrics=json.loads((folder/'metrics.json').read_text())
    speed=json.loads((folder/'ast_speed_attribution.json').read_text())
    needed={m['key'] for m in metrics if m['method'] in (
        'v5_structural','v5_structural_frequency','v5_structural_reward_greedy')}
    rows=defaultdict(list)
    for row in map(json.loads,gzip.open(folder/'per_formula.jsonl.gz','rt')):
        if row['key'] in needed and row['suite']!='stress_test':rows[(row['key'],row['suite'])].append(row)
    evidence=[]
    for capacity in (1024,2048):
        for scenario,suite in [('broad','fresh_test'),('specialized','extreme_domain_test')]:
            methods={m['method']:m for m in metrics if m['capacity']==capacity and m['scenario']==scenario and m['suite']==suite}
            rl=methods['v5_structural']
            for name in ('v5_structural_frequency','v5_structural_reward_greedy'):
                control=methods[name]
                evidence.append({'capacity':capacity,'scenario':scenario,'suite':suite,'left':'v5_structural','right':name,
                    'left_tokens_per_formula':rl['tokens_per_formula'],'right_tokens_per_formula':control['tokens_per_formula'],
                    'left_paper_tokens_per_byte':rl['document_macro_tokens_per_byte'],'right_paper_tokens_per_byte':control['document_macro_tokens_per_byte'],
                    'left_crossing':rl['independent_tree_crossing'],'right_crossing':control['independent_tree_crossing'],
                    'bootstrap':paired_bootstrap(rows[(rl['key'],suite)],rows[(control['key'],suite)])})
    (folder/'contribution_evidence.json').write_text(json.dumps({'same_ID_ast_speed':speed,'rl_same_atoms_controls':evidence,
        'scope':'Frozen models, same fit and atomization; no test-based method selection. Pairwise CIs descriptive, not adjusted for multiple comparisons.'},indent=2)+'\n')
    lines=['# AST-v4的速度来源与两个自研家族的贡献证据','',
        '当前结果支持数学公式专用分词系统的实用贡献，AST-v4的工程加速有直接同ID对照；v5的RL相对同先验控制的额外增益仍有限。论文层面的原创性和下游效用不能由综合名次单独证明。','',
        '## AST-v4为何快','',
        '在线路径是原始Python字符串→C词法扫描和AST/CST构建→命令/codepoint合法边界→词表trie→受约束最短路径→ID列表。每次调用都重新处理输入；完整计时包含前端与Python/C边界。词表加载和编译作为所有配置共同排除的初始化成本另记，评测统计/参考KaTeX树不属于任何模型的encode计时。','',
        '词表在加载时构造连续int32 trie数组，字节匹配按数组索引跳转。最短路径只做整数代价1+best[end]，每个候选区间的crossing状态在trie遍历时增量维护；不使用通用策略内核的softmax、对数、longest预扫描或n×maxlen跨节点矩阵。最坏扫描界仍与输入字节数和最大词条长度相关，不能声称对任何TeX严格线性。','',
        'CPython桥直接取得UTF-8视图及trie缓冲区，一次原生调用完成扫描、树、mask和DP，直接生成ID列表，减少ctypes调用、NumPy临时数组和中间树对象。支持算术优先级、上下标、分式/根号的轻量AST；其他语法使用明确CST回退，不是完整TeX宏执行、符号化简或通用数学语义解析。','',
        '训练阶段以结构合法区间学习/剪枝词表，上线固定词表下做受约束最短分词。因此AST-v4沿用AST-BPE家族，但当前推断不是逐轮执行BPE合并。1,024/2,048 ID的小词表和测试中的短公式也影响速度，尚未验证普通长文本和批量/多线程场景的优势。对成熟tokenizer的吞吐量差还含词表、正则预分词和接口实现差异；同一Llama 3词表在HF与Meta原版接口下速度不同，不能将实现快慢直接解释为算法或语言模型能力强弱。','',
        '以下七轮交错对照使用相同词表、同C前端、同硬约束、同3,465条新论文公式，逐token ID完全一致；避免把前端或词表变化归给工程优化。','',
        '|词表预算|融合CPython/C接口 MiB/s|同C前端经ctypes MiB/s|同C前端+通用DP MiB/s|融合相对ctypes|融合相对通用路径|',
        '|---:|---:|---:|---:|---:|---:|']
    for r in speed:
        x=r['encode_mib_s'];g=r['fused_bridge_speedup']
        lines.append(f"|{r['capacity']}|{x['fused_bridge']:.3f}|{x['ctypes_same_native_frontend']:.3f}|{x['generic_dp_same_native_frontend']:.3f}|{g['ctypes_same_native_frontend']:.2f}×|{g['generic_dp_same_native_frontend']:.2f}×|")
    current={m['method']:m for m in metrics if m['capacity']==1024 and m['scenario']=='broad' and m['suite']=='fresh_test'}
    lines+=['',f"上一轮AST-v4约30 MiB/s，本次含Meta原版Llama 3的全配置共同计时为{current['ast_v4']['encode_mib_s']:.2f} MiB/s。七轮原始时间表显示同一配置也有波动；当前没有测量系统负载/热状态来确定原因，不能把一次吞吐量作为硬件无关常数。同轮比较、原始重复读数与同ID实现对照应一起解释。",'',
        '上一轮旧版约1.6 MiB/s到约30 MiB/s的整套提升同时含Python前端改写、前端树定义变化和词表更新，不能全部作为纯工程等价加速。上表约3.35–3.47×接口收益和5.29–5.64×相对通用路径才有严格同ID证据。','',
        '## AST-v4可成立的贡献','',
        '贡献是将数学词法原子、结构区间约束、结构词表学习和高效完整编码组合为可部署的独立tokenizer，并给出同数据基准/等价实现证据。1024新论文命令完整98.89%、独立KaTeX crossing7.27%、27.691 token/式；2048为98.89%、9.22%、24.058 token/式。相同预算HF byte BPE分别为64.79%/23.45%/27.688以及76.29%/28.73%/23.839。AST取得结构优势，同时存在压缩代价。','',
        '不能将“AST、trie、动态规划或最短路径”本身写成原创发明。PathPiece已研究固定词表最少token分词，以及压缩和下游表现不一致：[Tokenization Is More Than Compression，EMNLP 2024](https://aclanthology.org/2024.emnlp-main.40/)。现有有限文献检索不足以证明本项目结构约束/词表设计从未有人提出。独立参考树仍是源语法跨度，不是数学语义金标准；自模型硬约束不意味着独立crossing为零。','',
        '## v5可成立与尚弱的贡献','',
        'v5离线学习合并偏好，导出固定词表和原版Rust BPE，上线不需要RL采样/策略网络。当前正式分支为训练内部contextual bandit的精确期望策略梯度，reward包含压缩、概率编码、词法、布局及独立结构credit；另保留完整轨迹RLOO研究分支。不是用户上传论文中token边界与语言模型损失联合优化的复现。','',
        '整套方法在两个预算两主场景中压缩领先正式基线，命令完整性新论文99.74%、专用跨领域98.19%，以固定原版BPE部署。它说明数学词法先验＋离线词表策略＋部署方式有效；命令原子和codec收益不能归给RL。仍有reward权重、atom预算、候选筛选和策略结构的人工选择，不属于完全没有人工介入。','',
        '真正衡量RL贡献需要同fit、同词法原子和同候选预算的频率与reward-greedy控制。下面使用按论文等权token/byte差（RL−控制）；负值对RL有利，区间为2,000次论文聚类bootstrap。','',
        '|预算|场景|控制|RL paper token/byte|控制 paper token/byte|差值95% CI|RL crossing%|控制 crossing%|',
        '|---:|---|---|---:|---:|---|---:|---:|']
    for e in evidence:
        ci=e['bootstrap']['ci95']
        lines.append(f"|{e['capacity']}|{e['scenario']}|{e['right']}|{e['left_paper_tokens_per_byte']:.6f}|{e['right_paper_tokens_per_byte']:.6f}|[{ci[0]:.6f}, {ci[1]:.6f}]|{100*e['left_crossing']:.3f}|{100*e['right_crossing']:.3f}|")
    lines+=['','1024新论文对频率控制的paper压缩优势区间含零，1024跨领域没有点估计优势；2048跨领域对频率控制有额外压缩收益，但同reward-greedy控制更短且crossing更低。2048新论文平均token/式略优于频率控制，论文等权压缩反而略差，不能混用两种聚合挑胜项。当前尚不能成立“RL普遍优于频率/贪心”或“免除人工设计”的强结论。','',
        '上传论文的贡献是用score-function方法与方差控制学习端到端离散token边界；我们的潜在差异在于数学多目标credit与离线固定词表导出。RL概念和两阶段思想本身不是充分原创性证据：[Dauncey & Wattenhofer，ICML 2026](https://proceedings.mlr.press/v306/dauncey26a.html)。','',
        '## 目前研究结论的强度','',
        '|层次|AST-v4|v5|','|---|---|---|',
        '|实际公式分词系统收益|结构保持与速度有明确证据，存在长度代价|压缩与命令完整性有明确整套方法证据，速度仍有优化空间|',
        '|工程贡献|同ID控制支持原生接口和专用DP加速|离线学策略、线上固定Rust BPE可运行，原子化转换开销已计入|',
        '|独立算法收益|结构设计有前景，须追加同词表约束开关/树构建消融分解归因|RL相对频率/贪心收益不稳定，目前是局部证据|',
        '|可主张的原创性|需更完整prior-art检索与约束机制论证|需证明学习策略超越同reward搜索/贪心，而非仅增加学习环节|',
        '|下游贡献|尚未测试|尚未测试|','',
        '下一阶段最关键的证据是冻结tokenizer后，在相同语言模型架构和资源协议下评估公式建模/任务表现；同时补充结构约束与树定义的归因消融，以及RL优于同reward简单控制的稳定跨域/跨种子证据。当前不据本征综合排名宣称下游推理提升或普遍最优。','',
        '证据：[同ID速度](../results/v6/full/ast_speed_attribution.json)、[贡献对照与区间](../results/v6/full/contribution_evidence.json)、[全模型直接比较](general_vs_math_v6.md)。']
    destination=ROOT/'docs/ast_speed_and_contributions_v6.md'
    destination.write_text('\n'.join(lines)+'\n');print(destination,flush=True)


if __name__=='__main__':main()
