"""Create four-metric tables and three publication-resolution figures."""
import csv,gzip,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('study',HERE/'run.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
O=s.OUT
NAMES={'ast_bpe':'Hard AST-BPE','soft_ast_bpe':'Soft AST-BPE','Qwen2.5-0.5B':'Qwen2.5','meta-llama/Meta-Llama-3-8B':'Llama 3','tiktoken_o200k_base':'o200k'}
COLORS={'ast_bpe':'#0072B2','soft_ast_bpe':'#D55E00','Qwen2.5-0.5B':'#009E73','meta-llama/Meta-Llama-3-8B':'#CC79A7','tiktoken_o200k_base':'#777777'}
METHODS=list(NAMES)
LABELS=['Fragmentation ↓ (%)','Cmd ↑ (%)','Cross preservation ↑ (%)','Num ↑ (%)']

def main():
 protocol=json.loads((O/'protocol.json').read_text());rows=[]
 for p in protocol['splits']:
  paths=list((O/p['name']).glob('*.json'))
  assert len(paths)==19,(p['name'],len(paths))
  for path in paths:
   r=json.loads(path.read_text());r.update(protocol=p['name'],seed=p['seed']);rows.append(r)
 assert len(rows)==228
 cols=['protocol','seed','method','requested_total_ids','actual_total_ids','formulas','papers']+s.METRICS+['test_ids_sha256']
 with (O/'all_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({c:r.get(c) for c in cols} for r in rows)
 s.write(O/'all_results.json',rows)
 # Five-model comparisons at every requested budget; the official models are reused,
 # never refitted or truncated. This expanded CSV makes all eight tables explicit.
 expanded=[]
 for p in protocol['splits']:
  for size in s.SIZES:
   for r in rows:
    if r['protocol']==p['name'] and (r['requested_total_ids'] is None or r['requested_total_ids']==size):expanded.append(dict(comparison_budget=size,**{c:r.get(c) for c in cols}))
 with (O/'five_model_comparisons_all_budgets.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['comparison_budget']+cols);w.writeheader();w.writerows(expanded)
 def select(prefix,method,size=1024):return [r for r in rows if r['protocol'].startswith(prefix) and r['method']==method and r['requested_total_ids']==(size if method in METHODS[:2] else None)]
 def values(prefix,method,metric,size=1024):return np.array([r[metric] for r in select(prefix,method,size) if r[metric] is not None])
 def cell(a):return 'NA' if not len(a) else f'{np.mean(a):.2f} ± {np.std(a,ddof=1):.2f}' if len(a)>1 else f'{a[0]:.2f}'
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
 for i,(prefix,title) in enumerate([('pde_id','PDE-only: unseen source papers'),('benchmark_id','Full benchmark: unseen source papers')]):
  for j,metric in enumerate(s.METRICS):
   ax=axes[i,j]
   for method,marker in [('ast_bpe','o'),('soft_ast_bpe','s')]:
    a=np.array([values(prefix,method,metric,size) for size in s.SIZES]);ax.plot(s.SIZES,a.mean(axis=1),marker=marker,markersize=4,color=COLORS[method],label=NAMES[method]);ax.fill_between(s.SIZES,a.min(axis=1),a.max(axis=1),alpha=.12,color=COLORS[method])
   ax.set_title(LABELS[j]);ax.set_xlabel('Total IDs (includes 3 controls)');ax.set_xticks([320,512,768,1024]);ax.set_ylim(-2,102);ax.grid(axis='y',alpha=.18)
   if j==0:ax.set_ylabel(title+'\nMetric (%)')
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
 fig.suptitle('Vocabulary-size sensitivity: independent fresh fits at every budget\nLines: mean across 3 source splits; shaded bands: split range (not a confidence interval)',fontsize=13)
 save(fig,'01_vocabulary_sensitivity')
 fig,axes=plt.subplots(2,4,figsize=(15,7.2),layout='constrained')
 for i,(prefix,title) in enumerate([('pde_id','PDE-only'),('benchmark_id','Full benchmark')]):
  for j,metric in enumerate(s.METRICS):
   ax=axes[i,j]
   for k,m in enumerate(METHODS):
    a=values(prefix,m,metric);mean=a.mean();ax.barh(k,mean,color=COLORS[m],height=.6,alpha=.8);ax.errorbar(mean,k,xerr=np.array([[mean-a.min()],[a.max()-mean]]),fmt='none',color='#222222',capsize=3,lw=1)
    ax.text(min(mean+1.5,96),k,f'{mean:.1f}',va='center',fontsize=9)
   ax.set_yticks(range(5),[NAMES[m] for m in METHODS] if j==0 else ['']*5);ax.invert_yaxis();ax.set_xlim(0,108);ax.set_xlabel(LABELS[j]);ax.grid(axis='x',alpha=.15)
   if j==0:ax.set_title(title,fontweight='bold')
 fig.suptitle('Five-tokenizer comparison: new models at 1,024 total IDs; official full vocabularies\nBars: mean across 3 source splits; whiskers: split range (not a confidence interval)',fontsize=13)
 save(fig,'02_five_tokenizers')
 fig,axes=plt.subplots(1,4,figsize=(16,5.1),layout='constrained')
 groups=['benchmark_id','benchmark_ood_algebra','benchmark_ood_geometry_topology','benchmark_ood_number_theory','benchmark_simple_to_complex'];tick=['ID','Algebra\nOOD','Geometry\nOOD','Number theory\nOOD','Simple →\ncomplex']
 for ax,metric,title in zip(axes,s.METRICS,LABELS):
  for k,m in enumerate(METHODS):
   a=[values(prefix,m,metric).mean() for prefix in groups];ax.plot(range(5),a,color=COLORS[m],marker=['o','s','^','D','v'][k],label=NAMES[m],lw=1.8,ms=5)
  ax.set_title(title);ax.set_xticks(range(5),tick,rotation=35,ha='right');ax.set_ylim(-2,102);ax.grid(axis='y',alpha=.18)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False)
 fig.suptitle('Generalization: 1,024-ID new tokenizers vs. unchanged mature tokenizers\nID and complexity points average 3 source splits; each OOD point is a separate held-out domain',fontsize=13)
 save(fig,'03_generalization')
 text=['# ZIP 两种 AST-BPE：从零训练的四指标实验','', '本报告只研究 ZIP 中的 Hard AST-BPE（`ast_bpe`）、Soft AST-BPE（`soft_ast_bpe`），与 Qwen2.5、Llama 3、o200k 对比。所有自研词表都在对应训练集上、按每个容量独立从零学习。未载入 ZIP 内的已有词表，未使用旧自研模型的训练检查点。','', '## 1. 指标与预先固定的比较方法','', '| 指标 | 定义 | 方向 |','|---|---|---|','| 破坏率／切分率 | $100\\sum_i(T_i-1)/\\sum_i(B_i-1)$，其中 $B_i$ 为 UTF-8 字节数，$T_i$ 为内容 token 数 | ↓ |','| Cmd | 内部没有 token 边界的 `\\[A-Za-z]+` 命令数／全部这类命令数 ×100 | ↑ |','| Cross 保持率 | $100(1-\\mathrm{partial\\ crossing\\ tokens}/\\mathrm{all\\ tokens})$，参考为冻结的独立 KaTeX 语法跨度 | ↑ |','| Num | 内部没有 token 边界的 `[0-9]+(?:\\.[0-9]+)?` 数字词素数／全部这类数字词素数 ×100 | ↑ |','', '统计都使用先汇总分子、分母再求比率；没有命令或数字时是 NA，不记为100。Cross 是旧 crossing rate 的补数，与旧表的方向相反。判定部分跨越：token `[a,b)` 与参考跨度 `[s,e)` 满足 `a<s<b<e` 或 `s<a<e<b`；包含、被包含、端点重合均不算 crossing。','', '**“破坏率”在这里严格测量切分密度，不直接等价于数学语义受损。** 例如3字节 `abc` 切为3个 token，切分率为100%；切为1个 token，切分率为0%。逐字节切分虽然 Cmd/Num 可能很差，却几乎不会部分跨越树跨度，Cross 可以得到100%。因此四指标不保证反向单调关系，不能修改定义以追求这种关系。','', '## 2. 数据、模型及公平性','', '现有冻结 catalog 共56,000条：严格 PDE 356条，其余数学55,644条。没有为本轮更改数据或分类。严格 PDE 来自20个来源组，不等于“PDE论文内的所有公式”。全量 benchmark 包含13个领域。','', '词表8档：320、384、448、512、640、768、896、1024。每档总ID含 PAD/EOS/BOS 三个控制ID；指标仅计算内容 token，不添加这些控制ID。两种新模型使用完全相同的训练公式，逐档独立训练。三个成熟分词器保留官方完整词表，因此本实验是实际系统质量比较，**不是同词表预算、同预训练数据的算法因果比较**。成熟模型可能见过公开数据，无法证明其预训练阶段未接触这些公式。','', '| 模型 | 来自何处／算法 | 实际ID空间 |','|---|---|---|','| Hard AST-BPE | ZIP 原始 `HardASTBPE`；字节初始化；结构兼容的频率合并；编码按训练合并顺序并施加硬限制 | 每档实测记录，含3控制ID |','| Soft AST-BPE | ZIP 原始 `train_one`；训练侧命令／Unicode原子 + 官方HF BPE训练；结构感知动态规划；固定 tree=.25、boundary=.30 | 每档实测记录，含3控制ID |']
 for m in METHODS[2:]:text.append(f"| {NAMES[m]} | 原项目固定官方代码／词表，原样使用 | {select('benchmark_id',m)[0]['actual_total_ids']:,} |")
 text += ['', '为避免无关的 EqGPT 数据依赖，只从原文件解析并原样执行 `train_one` 函数体；其依赖实现来自逐字节一致的 ZIP 源码快照。没有改写分词算法或训练函数。源码与 ZIP 哈希见 `zip_provenance.json`。','', '所有模型输入相同原字符串；不截断、不归一化、不按模型剔除公式；完整执行逐条 encode/decode 和字节重建断言。训练／验证／测试来源组隔离，且剔除训练侧与验证或测试侧相同规范公式／模板。验证集不用于建词表或选择参数，主比较容量预先固定为1024。','', '| 划分 | 训练 | 验证（未拟合） | 测试 | 训练侧去重剔除 |','|---|---:|---:|---:|---:|']
 for p in protocol['splits']:text.append(f"| {p['name']} | {len(p['train']):,} | {len(p['dev']):,} | {len(p['test']):,} | {p['purged_train']} |")
 text += ['', 'PDE与全量ID各3个来源隔离划分；OOD分别完整留出代数、几何拓扑、数论领域，并从训练排除与该测试领域共享来源的公式；复杂度实验只使用简单公式训练，在其他来源的复杂公式上测试。重复划分可能重叠，不能当成独立采样，均值±标准差只描述划分敏感性，不当作置信区间。每个单独实验还提供500次来源组聚类bootstrap的95%区间。复杂度实验同时包含来源变化，不能单独把差异因果归结为复杂度。','', '## 3. 词表规模的影响','', '![8档词表规模曲线](figures/01_vocabulary_sensitivity.png)','', 'Soft AST-BPE 在小词表时优先把训练中出现的命令与Unicode符号加入原子字母表（最多384个额外原子，另有256个字节原子），剩余预算才用于BPE合并。因此容量变化同时影响原子覆盖与合并数，可能出现明显拐点；这些是各容量独立拟合的模型，不是从1024词表简单截断。完整逐划分、逐容量结果见 `all_results.csv`；五模型在全部8档容量下的显式对照表见 `five_model_comparisons_all_budgets.csv`，其中通用模型结果不随自研容量重训。','']
 for prefix,label in [('pde_id','纯PDE'),('benchmark_id','全量 benchmark')]:
  text += [f'### {label}：两种方法的容量曲线数值','', '| 总ID | 方法 | 破坏率↓ | Cmd↑ | Cross↑ | Num↑ |','|---:|---|---:|---:|---:|---:|']
  for size in s.SIZES:
   for m in METHODS[:2]:text.append('| '+str(size)+' | '+NAMES[m]+' | '+' | '.join(cell(values(prefix,m,k,size)) for k in s.METRICS)+' |')
 text += ['', '## 4. 五模型对比（1024-ID新模型）','', '![五模型四指标](figures/02_five_tokenizers.png)','']
 for prefix,label in [('pde_id','纯PDE未见公式'),('benchmark_id','全量 benchmark 未见来源')]:
  text += [f'### {label}','', '| 方法 | 破坏率↓ | Cmd↑ | Cross↑ | Num↑ |','|---|---:|---:|---:|---:|']
  for m in METHODS:text.append('| '+NAMES[m]+' | '+' | '.join(cell(values(prefix,m,k)) for k in s.METRICS)+' |')
 text += ['', '## 5. 泛化实验','', '![泛化四指标](figures/03_generalization.png)','','每个场景内部五模型使用相同测试公式；跨领域点的公式分布、训练集合不同，折线只呈现场景表现，不能直接把两点相减解释为纯粹的泛化损失。OOD每次完整留出一个领域，并重新拟合两种新模型。未知指未进入本轮词表拟合；无法排除通用tokenizer的历史预训练接触。','']
 for prefix,label in [('benchmark_ood_algebra','从未参与拟合的代数领域'),('benchmark_ood_geometry_topology','从未参与拟合的几何拓扑领域'),('benchmark_ood_number_theory','从未参与拟合的数论领域'),('benchmark_simple_to_complex','仅简单公式训练 → 未见来源的复杂公式')]:
  text += [f'### {label}','', '| 方法 | 破坏率↓ | Cmd↑ | Cross↑ | Num↑ |','|---|---:|---:|---:|---:|']
  for m in METHODS:text.append('| '+NAMES[m]+' | '+' | '.join(cell(values(prefix,m,k)) for k in s.METRICS)+' |')
 text += ['', '## 6. 按指标比较与解释','','1024-ID 常规未见公式测试中，Soft AST-BPE 的切分率最低、Cmd最高；Hard AST-BPE 的Cross保持率最高。两种方法的Num都没有全面超过Llama 3和o200k。Soft的Cross均值优势较小，在几何、数论OOD及部分PDE划分中，相对Qwen的配对区间包含零，不能宣称稳定的结构优势。数值相同的模型视为并列。','', '以下使用同一组测试划分的均值作描述性排序，不人为指定综合权重，不把微小均值差异当作显著优胜。指标方向已固定。另提供 `paired_comparisons_1024.csv`：同一来源组抽样下，逐指标计算新模型相对基准的配对差值及1000次聚类bootstrap的95%区间，差值为正表示新模型更好；这些是条件于各划分、未做多重比较校正的探索性区间。','']
 ranks={}
 for prefix,label in [('pde_id','PDE'),('benchmark_id','全量ID'),('benchmark_ood_algebra','代数OOD'),('benchmark_ood_geometry_topology','几何OOD'),('benchmark_ood_number_theory','数论OOD'),('benchmark_simple_to_complex','简单→复杂')]:
  text.append(f'**{label}**')
  ranks[prefix]={}
  for k in s.METRICS:
   mm={m:float(values(prefix,m,k).mean()) for m in METHODS};ordered=sorted(mm,key=lambda m:mm[m],reverse=k!='fragmentation_pct');ranks[prefix][k]=[dict(method=m,value=mm[m]) for m in ordered]
   chunks=[]
   for m in ordered:chunks.append(f'{NAMES[m]} ({mm[m]:.2f})')
   text.append('- '+k+'：'+'；'.join(chunks))
  text.append('')
 text += ['为什么通用 tokenizer 的 Cmd 会很低：指标要求完整保留反斜杠在内的命令内部边界，例如 `\\partial` 切为 `\\` 与 `partial` 就失败，即使字母拼写没有再切。这是预先指定的严格命令完整性，不是模型不理解偏导，也不能由此推出下游任务差。Num 在单字符数字很多的测试中容易高分，必须同时看数字词素数量和长度分布，不能把接近100自动解读为多位数处理优异。','', '## 7. 审查和交付','', '三轮审查详见 `audit_round_1.json`、`audit_round_2.json`、`audit_round_3.json`。第一轮核对ZIP来源、隔离划分和指标手算；第二轮比较硬AST的原生／Python参考编码，比较软AST的原生／独立动态规划目标值，并从零重训核对端点容量；第三轮独立汇总全部原始计数并核对共享测试ID、分母及检查点。','']
 for i in (1,2,3):
  a=json.loads((O/f'audit_round_{i}.json').read_text());assert a['status']=='PASS' and a['scope']=='complete';text.append(f"- 第{i}轮：{a['status']}。")
 a=json.loads((O/'audit_round_1.json').read_text());text+=['', '数字词素长度分布（所有catalog，只用于说明指标难度）：`'+json.dumps(a['number_length_counts'],ensure_ascii=False)+'`。','', 'PDE数字词素长度分布：`'+json.dumps(a['pde_number_length_counts'],ensure_ascii=False)+'`。','', '复现命令：','```bash','.venv/bin/python scripts/zip_ast_20261021/run.py','.venv/bin/python scripts/zip_ast_20261021/audit.py 1','.venv/bin/python scripts/zip_ast_20261021/audit.py 2','.venv/bin/python scripts/zip_ast_20261021/audit.py 3','.venv/bin/python scripts/zip_ast_20261021/paired.py','.venv/bin/python scripts/zip_ast_20261021/report.py','```','', '所有 per-formula `.jsonl.gz` 包含可复算分子分母；模型文件和训练ID哈希在各划分 `models/`。图提供PNG、PDF和SVG。原数据、旧模型、PPT和下游代码均未由本轮修改。']
 (O/'REPORT_ZH.md').write_text('\n'.join(text)+'\n');s.write(O/'metric_rankings.json',ranks)
 s.write(O/'completion.json',dict(status='complete',protocols=12,independently_fitted_tokenizers=192,evaluations=228,expanded_five_model_comparison_rows=len(expanded),metrics=s.METRICS,audits=[json.loads((O/f'audit_round_{i}.json').read_text())['status'] for i in (1,2,3)]))
 print('Report and figures complete:',O)

def save(fig,name):
 folder=O/'figures';folder.mkdir(exist_ok=True)
 for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
 plt.close(fig)
if __name__=='__main__':main()
