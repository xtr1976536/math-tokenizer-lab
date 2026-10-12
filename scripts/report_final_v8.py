"""Complete final intrinsic experiment and pretest downstream selection."""
import json
from math_tokenizer.corpus import ROOT
try:
    from scripts.report_v7 import NAMES
except ModuleNotFoundError:
    from report_v7 import NAMES


NAMES=dict(NAMES,ast_boundary='AST 参数边界路径',parameter_rl='参数级终端RL',random_search='固定随机搜索',cem_search='交叉熵搜索',no_update_v8='关闭更新')


def main():
    folder=ROOT/'results/v8/full'
    ranks=json.loads((folder/'rankings.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
    audit=json.loads((folder/'fairness_audit.json').read_text());manifest=json.loads((ROOT/'data/releases/v8/manifest.json').read_text())
    decision=json.loads((ROOT/'results/v8/controlled_terminal/decision.json').read_text())
    final=json.loads((ROOT/'models/final_intrinsic/registry.json').read_text())
    speed=json.loads((ROOT/'results/v8/training_speed.json').read_text())
    verification=json.loads((ROOT/'results/v8/verification.json').read_text())
    effects=json.loads((folder/'multimetric_paired_effects.json').read_text())
    assert verification['all_passed'] and audit['all_passed']
    lines=['# v8：进入下游前的最终本征分词器实验','',
        '本轮先写定RL实用晋升门槛、相同终端奖励与评分预算、AST边界开发网格、最终比较和下游角色，再评分新增封存测试。完整保留失败种子、搜索控制及历史基准。最终结论针对当前独立分词器，不能推广为所有RL或语言模型内端到端RL没有作用。','',
        f"新测试包含{manifest['formulas']:,}条公式，{manifest['papers_with_accepted_formulas']}篇论文贡献，{manifest['papers_downloaded']}篇许可核验源码被获取，来源领域外子集{manifest['splits']['extreme_domain_test']:,}条。预先目标32篇，元数据候选中实际取得{manifest['papers_downloaded']}篇；领域覆盖有限，未按模型分数补样本。固定2020年前版本、源码许可、宏变换、领域和simple/complex证据均保留，历史人工来源仍是年代推断。",'',
        f"正式{verification['configurations']}种配置、{verification['metric_records']}组指标、{verification['formula_rows']:,}行结果；九种成熟tokenizer全部同原式评测。除了新测试，还统一重跑暴露的v7/v6自然和来源领域外测试、521式压力集。共{audit['total_rounds']}轮协议/结果审查、{verification['tests_passed']}项测试通过。七轮交错公共encode/decode，包括codec和前端。旧测试报告不称新未见泛化。",'',
        '## RL：同奖励、同预算的实用去留','',
        '新候选采样完整24维合并策略，用相同两个训练任务上实际导出的词表终端回报进行Gaussian score-function更新，避免把几百次动作score相加。固定随机搜索和CEM搜索共享同初始化、候选特征、目标容量及多目标回报；每算法17个候选×2任务=34次终端评分，均按训练内回报选择最佳已评分候选。关闭更新单列，不伪装成同样计算量。允许信息和评分次数一致，实际耗时并不完全相同。','',
        '三个按论文分离的来源划分×五训练种子；这些属于以前可访问的开发来源。晋升要求相对两种搜索均至少0.5%平均utility改善、15组中至少12胜、种子重采样区间支持改善，并满足压缩/边界完整性非劣约束。区间条件于三次可能重叠来源划分，不能把十五组当成完全独立来源实验。','',
        '|对照|utility相对改善↑|15组胜数|种子聚类95%区间（绝对utility改善）|核心非劣界|晋升|',
        '|---|---:|---:|---|---|---|']
    for name,p in decision['comparisons'].items():
        ci=p['seed_cluster_ci95']
        lines.append(f"|{NAMES[name]}|{100*p['relative_improvement_mean']:.3f}%|{p['winning_pairs']}/15|[{ci[0]:.6f}, {ci[1]:.6f}]|{'通过' if p['all_noninferiority_guards_passed'] else '未通过'}|{'通过' if p['promotion_passed'] else '未通过'}|")
    lines+=['',
        ('RL通过开发晋升并保留为下游核心候选。' if decision['rl_promoted'] else 'RL没有通过固定的实用晋升门槛，最终生产及下游核心名单去掉RL。保存旧RL及本轮所有候选，作为诊断和负面结果复现。不是从磁盘抹去失败证据，也不是证明所有RL无效。'),'',
        '![同奖励同预算的RL多种子结果](../results/v8/figures/rl_controlled_seeds.png)','',
        '四个完整训练场景另各训练三个种子，实际目标容量1024/2048，不再只从较小容量任务迁移；三算法共享相同奖励和34次评分。正式新测试全部报告，不选幸运seed。','',
        '## AST：参数边界与压缩取舍','',
        '新路径的代价为token数＋native partial-crossing代价＋被token跨过的源码brace内容端点代价。它只读取原文，推断不加载KaTeX/pylatexenc参考。边界权重在共同旧dev的固定12项网格选择，所有场景选tree=.25、boundary=.3；独立pylatexenc作为不同实现的语法诊断，仍非专家数学语义gold。字典和原子与原版同原子BPE完全一致。','',
        '|预算|场景|方法|paper token/byte↓|KaTeX crossing%↓|独立参数边界召回%↑|命令完整%↑|编码MiB/s↑|',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            board=ranks['within_scenario'][f'{capacity}/{scenario}']['metrics']
            for name in ('ast_v4_parent','v5_parent','official_atomic_bpe','ast_lexicographic','ast_boundary','parameter_rl','random_search','cem_search'):
                m=board[name]
                lines.append(f"|{capacity}|{scenario}|{NAMES[name]}|{m['document_macro_tokens_per_byte']:.5f}|{100*m['independent_tree_crossing']:.2f}|{100*m['independent_scope_boundary_recall']:.2f}|{100*m['command_integrity']:.2f}|{m['encode_mib_s']:.2f}|")
    lines+=['',
        '新版必须同时看compression、KaTeX crossing与另一组边界指标。低crossing或高边界召回各自都可能被退化分词获得；不能据其中一项证明整体数学语义更好。全量独立指标、解析覆盖、局部编辑、平滑敏感性、Rényi效率、词表使用及长尾长度均在metrics.json。','',
        '![新封存测试的压缩与边界取舍](../results/v8/figures/ast_tradeoffs.png)','',
        f"按论文聚类{effects['resamples']:,}次重采样，主要比较×六核心计数指标共{effects['family_size']}项，报告Bonferroni家族95%区间。下面均为AST边界路径减同原子BPE；长度与crossing负值有利，边界recall正值有利。区间条件于保存的词表及已观察训练seed。",'',
        '|预算/场景|paper token/byte差：家族95%CI|参数边界召回差pp：家族95%CI|crossing差pp：家族95%CI|',
        '|---|---|---|---|']
    for capacity in (1024,2048):
        for scenario in ('broad','specialized'):
            p=effects['comparisons'][f'{capacity}/{scenario}/ast_boundary-vs-official_atomic_bpe']['metrics']
            cells=[]
            for key,scale in [('document_macro_tokens_per_byte',1),('independent_scope_boundary_recall',100),('independent_tree_crossing',100)]:
                v=p[key];ci=v['familywise_ci95'];cells.append(f"{scale*v['difference_left_minus_right']:+.5f} [{scale*ci[0]:+.5f}, {scale*ci[1]:+.5f}]")
            lines.append(f"|{capacity}/{scenario}|"+'|'.join(cells)+'|')
    lines+=['',
        '四场景压缩差的家族区间均支持改善；边界与crossing的部分区间跨零，不能将全部点估计改善说成稳定显著优势。极端来源域的论文数较少，区间更宽，来源领域标签也不等同于完全不同数学概念。','',
        f"完整语料C++索引在{speed['formulas']}条fit、1024目标容量、五轮交错完整词表构建对照上加速{speed['median_rollout_speedup']:.2f}倍，词表/合并/策略score完全一致。该计时包括codec、特征、计数及原版BPE构建，排除终端奖励评估；不解释成学习收益或完整RL同倍加速。",'',
        '## 去掉RL的替代核验','',
        '同原子原版BPE作为最简单的非RL生产候选。它是原版BPE加数学词法原子的对照/替代，不包装成原创RL算法。评分前固定的容忍界：压缩相对1%、BPB相对2%、命令/数字/独立边界召回与crossing绝对1个百分点、编码速度降低10%。满足点估计界与证明统计等价分开。','',
        '|预算/场景|压缩相对退化%↓|BPB相对退化%↓|参数边界召回损失pp↓|crossing增加pp↓|速度相对损失%↓|全部点估计界|压缩家族CI非劣|',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for label,e in final['non_RL_replacement_effects'].items():
        lines.append(f"|{label}|{100*e['compression_relative_regression']:.3f}|{100*e['fit_bpb_relative_regression']:.3f}|{100*e['scope_recall_loss']:.3f}|{100*e['crossing_increase']:.3f}|{100*e['encode_relative_loss']:.3f}|{'通过' if e['point_estimate_guards_passed'] else '未通过'}|{'支持' if e['compression_familywise_ci95_within_margin'] else '不支持'}|")
    lines+=['',
        '完整命令/数字差、配对压缩区间及每一容忍界见non_RL_replacement.json；多指标普通与家族调整区间见multimetric_paired_effects.json。某些界若未通过，不能声称去掉RL后各指标完全保持。工程去留依据无稳定学习收益和明确角色；旧v5留为下游诊断消融，避免掩盖结构取舍。','',
        '已冻结AST边界分支与父v5的替代差另外作描述性核验，见AST_non_RL_replacement_descriptive.json。该比较未追加为主要预注册假设，不用它重新选择模型或宣称统计等价。两者的词表及分词规则不同，结果支持完整非RL替代系统的实用指标，不能将全部差异归因于单独删除RL。','',
        '|AST替代父v5：预算/场景|压缩相对退化%↓|参数边界召回损失pp↓|crossing增加pp↓|所有核心点估计容忍界|',
        '|---|---:|---:|---:|---|']
    for label,e in final['frozen_AST_vs_RL_parent_descriptive'].items():
        lines.append(f"|{label}|{100*e['compression_relative_regression']:.3f}|{100*e['scope_recall_loss']:.3f}|{100*e['crossing_increase']:.3f}|{'通过' if e['point_estimate_guards_passed'] else '未通过'}|")
    lines+=['',
        '## 完整排名','',
        '综合分沿用冻结权重：压缩30%、编码25%、KaTeX crossing20%、命令10%、数字5%、fit BPB10%；broad和specialized各50%。百分位综合排序依赖参赛集合，不能当作算法贡献或下游结论。另提供全部单指标排名、Pareto和速度权重敏感性。']
    for capacity,c in ranks['combined'].items():
        lines+=['',f'### {capacity}预算与成熟原版词表','',
                '|综合位置|方法|综合分↓|广域压缩名次|严格跨来源域压缩名次|','|---:|---|---:|---:|---:|']
        for rank,name in enumerate(c['ranking'],1):
            a=ranks['within_scenario'][f'{capacity}/broad']['metric_ranks']['document_macro_tokens_per_byte'][name]
            b=ranks['within_scenario'][f'{capacity}/specialized']['metric_ranks']['document_macro_tokens_per_byte'][name]
            lines.append(f"|{rank}|{NAMES.get(name,name)}|{c['scores'][name]:.4f}|{a}|{b}|")
    lines+=['','## 冻结的下游候选','',
        '名单角色在首次v8评分前冻结。相同容量研究组使用同训练信息，成熟模型保留其实际官方词表，后续须披露embedding/softmax参数及字节/算力预算，不能把不同容量误称严格等参。新旧分词器均可独立加载，无语言模型集成或下游训练。','',
        '|角色|方法|用途|','|---|---|---|',
        '|数学核心|AST参数边界路径、原版硬AST、同原子原版BPE|边界/压缩折中、严格结构、非RL数学原子对照|',
        '|相同容量控制|原版byte BPE、SentencePiece Unigram|传统算法及Unigram控制|',
        '|成熟通用核心|Qwen2.5、Llama 3、o200k|不同主流官方分词机制，实际词表容量单列|',
        '|结构必要消融|AST两级长度优先路径、作者PathPiece|区分边界代价、最短路径与实现贡献|',
        '|扩展通用|Qwen3、DeepSeek-V3、Phi-4、Mistral-Nemo、GPT-2、cl100k|兼容/部署参考，完整本征结果已包含|',
        '|学习诊断|父v5、终端RL、随机搜索、CEM、关闭更新|RL必要性复核，核心RL是否纳入依预定开发门槛|','',
        '每个角色准确检查点、sha256、容量/场景及三seed中的固定部署seed17，见models/final_intrinsic/registry.json；入口：from math_tokenizer.final_models import load_final。训练采用原版官方BPE的非RL路线，不隐含调用RL生成生产词表。','',
        '## 公平性与限制','',
        '评分前四轮检查：来源及同reward/34评分预算；公共接口异常输入与所有ID容量；同字典/原子和穷举/数值梯度；固定模型、原版第三方及完整测试。评分后四轮：每组全量输入/聚合；每个公共encoder逐式重放原文/payload/参考指标；七轮计时与作者信息指标；同约束路径原理核验。另复核学习与搜索所有日志和三来源重划分保存模型。','',
        '各方法实际wall-clock训练耗时和历史超参搜索不同。九种成熟tokenizer预训练污染未知，近期新来源并不意味着概念完全不重合。新测试覆盖有限且领域不均。源码边界指标已进入本轮AST开发，所以独立实现不等于完全未针对它优化。专家双审完成0，80条新测试待审材料已准备，未执行第二硬件或独立团队复现。没有下游任务结果。','',
        '本轮作为最后一次本征大规模封存实验；v8测试现在已暴露。后续重点转为固定候选的下游协议和实验，不能继续用此测试挑新的分词配置。','',
        '复现入口：sh scripts/run_v8_replay.sh。协议：docs/v8_preregistered_protocol.md；主结果：results/v8/full/metrics.json、all_main_metrics.csv、rankings.json、multimetric_paired_effects.json；逐式结果：per_formula.jsonl.gz；核验：results/v8/verification.json。']
    (ROOT/'docs/v8_final_intrinsic_results.md').write_text('\n'.join(lines)+'\n')
    print('Final report written',flush=True)


if __name__=='__main__':main()
