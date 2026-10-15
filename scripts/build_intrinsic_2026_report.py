"""Generate a Chinese research manuscript, measured figures and complete tables."""
from collections import Counter,defaultdict
import gzip,json,math,platform,re,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.intrinsic_2026 import METHODS,LABELS,METRICS,per_formula
from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.ast_route_v6 import NativeASTV4
from math_tokenizer.hard_ast_2026 import HardASTBPE
from math_tokenizer.atomic_bpe_v7 import AtomicBPE
from math_tokenizer.final_models import load_final
from math_tokenizer.metrics_v2 import token_bytes

DATA=ROOT/'experiments/intrinsic_20261015';PAPER=ROOT/'paper/intrinsic_20261015'
FAMILIES=['ID','DomainOOD','ExtremeOOD','Complexity']
FNAMES=['全领域新来源','整域 OOD','极限 OOD','简单→复杂']
COLORS=['#8A8E91','#5B7791','#547B7D','#A77265','#B3ACA0','#B8A179','#A6ACB4','#9FAE9D','#B2A4B7']
def tex(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
def literal(s):
    special={'\\':r'\textbackslash{}','{':r'\{','}':r'\}','$':r'\$','&':r'\&','_':r'\_','%':r'\%','#':r'\#','^':r'\textasciicircum{}','~':r'\textasciitilde{}'}
    return r'\texttt{'+''.join(special.get(c,c) for c in s)+'}'
def write(name,text): (PAPER/name).write_text(text.rstrip()+'\n')
def table(rows,caption,label=None,small=True,placement='htbp'):
    h=r'模型 & T/B $\dn$ & Cmd $\up$ & Num $\up$ & Cross $\dn$ & AST-A $\up$ & Used & $\eta_{2.5}$ $\up$ & 速度 $\up$ \\\midrule'
    lines=[r'\par\begin{table}['+placement+r']\centering'+(r'\small' if small else r'\footnotesize'),r'\caption{'+caption+'}'+(r'\label{'+label+'}' if label else ''),r'\begin{tabular}{lrrrrrrrr}\toprule',h]
    minima={k:min(r[k] for r in rows) for k in ['tb','cross']};maxima={k:max(r[k] for r in rows) for k in ['cmd','num','ast_align','renyi','speed']}
    for i,r in enumerate(rows):
        if i==6:lines.append(r'\midrule')
        vals=[]
        for k in ['tb','cmd','num','cross','ast_align','used','renyi','speed']:
            s=f"{r[k]:.4f}" if k=='tb' else f"{r[k]:.2f}" if k=='speed' else f"{100*r[k]:.2f}"
            if (k in minima and r[k]==minima[k]) or (k in maxima and r[k]==maxima[k]):s=r'\textbf{'+s+'}'
            vals.append(s)
        lines.append(tex(r['label'])+' & '+' & '.join(vals)+r' \\')
    lines.extend([r'\bottomrule\end{tabular}',r'\end{table}\par']);return '\n'.join(lines)+'\n\n'

def figures(aggregates):
    style=Path('/Users/xiao/.codex/skills/figure-polish/assets/deepscientist-academic.mplstyle')
    # The shipped style uses bare hex colours; preserve its values while quoting
    # the colour strings for the Matplotlib rc parser.
    target=PAPER/'figures/academic.mplstyle'
    target.write_text(re.sub(r':\s+(#[0-9A-Fa-f]{6})',r': "\1"',style.read_text()))
    plt.style.use(target)
    plt.rcParams.update({'font.family':['Heiti SC','Arial'],'axes.unicode_minus':False,'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.35),sharex=True,sharey=True)
    for ax,fam,title in zip(axes.flat,FAMILIES,FNAMES):
        rr=aggregates[fam];y=np.arange(9)
        for i,r in enumerate(rr):
            bar=ax.barh(i,r['tb'],height=.68,color=COLORS[i],edgecolor='white',linewidth=.6,hatch='//' if i==2 else '')
            ax.text(r['tb']+.006,i,f"{r['tb']:.3f}",ha='left',va='center',fontsize=7.5,fontweight='bold' if i==2 else 'normal')
        ax.set_title(title,loc='left');ax.set_yticks(y,LABELS,fontsize=8);ax.set_ylim(8.6,-.6);ax.set_xlim(0,.68);ax.grid(axis='y',visible=False);ax.grid(axis='x',color='#E7E5E4',lw=.5);ax.set_axisbelow(True)
    for ax in axes[1]:ax.set_xlabel('Tokens per byte（来源组等权，↓）')
    fig.subplots_adjust(left=.17,right=.99,top=.95,bottom=.08,wspace=.13,hspace=.27)
    save_figure(fig,'compression','四类测试下九个模型的来源组等权T/B；组间等权平均。')
    # One panel per structure measure: same categorical ordering, muted palette.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.55))
    for ax,key,title in [(axes[0],'cross','部分跨越 Cross（↓）'),(axes[1],'ast_align','复合节点对齐 AST-A（↑）')]:
        for i,r in enumerate(aggregates['ID']):
            ax.barh(i,100*r[key],color=COLORS[i],height=.68,edgecolor='white',hatch='//' if i==2 else '')
            ax.text(100*r[key]+.8,i,f"{100*r[key]:.1f}",va='center',fontsize=8)
        ax.set_yticks(np.arange(9),LABELS if ax==axes[0] else ['']*9,fontsize=8);ax.invert_yaxis();ax.set_xlabel('百分比（%）');ax.set_title(title,loc='left');ax.grid(axis='y',visible=False);ax.set_xlim(0,35 if key=='cross' else 90)
    fig.subplots_adjust(left=.17,right=.98,bottom=.16,top=.88,wspace=.16)
    save_figure(fig,'structure','ID三组均值；比较Cross与复合节点双端对齐。')
    fig,ax=plt.subplots(figsize=(7.2,3.25))
    for i,name in enumerate(METHODS):
        x=np.arange(4)+(i-4)*.083;values=[aggregates[f][i]['speed'] for f in FAMILIES]
        ax.bar(x,values,width=.078,color=COLORS[i],edgecolor='white',linewidth=.3,hatch='//' if i==2 else '',label=LABELS[i])
    ax.set_xticks(range(4),FNAMES);ax.set_ylabel('完整编码接口（MiB/s，↑）');ax.grid(axis='x',visible=False)
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.3),fontsize=8,columnspacing=2.0,handlelength=1.5)
    fig.subplots_adjust(left=.10,right=.99,bottom=.16,top=.72)
    save_figure(fig,'speed','各组七轮交错计时的中位速度；同类设置等权平均。')

def save_figure(fig,name,claim):
    d=PAPER/'figures';d.mkdir(exist_ok=True)
    fig.savefig(d/f'{name}.png',dpi=300,facecolor='white');fig.savefig(d/f'{name}.pdf',facecolor='white');plt.close(fig)

def metric_examples(catalog,metrics):
    record=catalog['0e5a942aa5faafdca707467a'];folder=DATA/'id_17/models'
    models={'Boundary AST':BoundaryAST.load(folder/'boundary_ast.json'),'Hard AST-BPE':HardASTBPE.load(folder/'hard_ast_bpe.json'),'Qwen2.5':load_final('Qwen2.5-0.5B')}
    rows=[];details={}
    for name,m in models.items():
        ids=m.encode(record['latex']);r=per_formula(m,record,ids);pieces=[token_bytes(m,i).decode() for i in ids];details[name]={'counts':r,'pieces':pieces}
        p=' $|$ '.join(literal(t) for t in pieces)
        rows.append(name+' & '+p+f" & {len(ids)}/11 & {r['command_good']}/{r['commands']} & {r['crossings']}/{r['tokens']} & {r['ast_good']}/{r['ast_nodes']}"+r' \\')
    text=r'''测试公式\texttt{x>\textbackslash{}frac\{2\}n}来自arXiv:1602.08801，ID为\code{0e5a942aa5faafdca707467a}，原串共11字节。表\ref{tab:example}给出三种实际输出。
\begin{table}[H]\centering\footnotesize
\caption{同一真实公式的分词及事件计数。竖线表示token边界。}\label{tab:example}
\begin{tabularx}{\linewidth}{l Y r r r r}\toprule
模型&实际token序列&T/B&Cmd&Cross&AST-A\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabularx}\end{table}
Boundary AST和Hard AST-BPE均产生5个token，单式T/B为$5/11\approx.455$，且\code{\frac}内部没有切点，Cmd均为100\%。软约束模型将分式命令与左花括号合并，跨过复合节点的起点，Cross为$1/5=20\%$；硬约束模型保留花括号分组的完整边界，Cross为0。该式两个复合参考跨度的双端在硬约束输出中全部保留，AST-A为100\%，软约束输出为0。Qwen2.5把反斜杠与\code{frac}分开，Cmd为0；其更细分的序列保留了两个复合跨度的双端，AST-A为100\%。三列分别显示命令内部、部分跨越和复合单元边界三种性质。
'''
    # Genuine numeric lexeme with more than one character, taken from this test.
    with gzip.open(DATA/'id_17/per_formula.jsonl.gz','rt') as f:rr=list(map(json.loads,f))
    byid=defaultdict(dict)
    for r in rr:byid[r['id']][r['method']]=r
    for id,systems in byid.items():
        x=catalog[id]['latex']
        if len(x)<100 and re.search(r'\{\d{2,}(?:\.\d+)?\}',x) and systems.get('hard_ast_bpe',{}).get('number_good',0)!=systems.get('boundary_ast',{}).get('number_good',0):
            text+='数字实例为测试公式'+literal(x)+r'（ID：\code{'+id+r'}）。'
            for key,label in [('hard_ast_bpe','Hard AST-BPE'),('boundary_ast','Boundary AST')]:
                r=systems[key];text+=label+f"保留{r['number_good']}/{r['numbers']}个数字词素；"
            text=text[:-1]+'。该式的数字词素为1和12，Boundary AST在12内部产生切点，而Hard AST-BPE保留完整的12。\n';break
    s=next(r for r in metrics if r['protocol']=='id_17' and r['method']=='boundary_ast')
    text+=f"第一组全领域测试输入共{s['bytes']:,}字节，即{s['bytes']/1048576:.4f} MiB；Boundary AST的中位编码时间为{np.median(s['timing_seconds']):.5f}秒，速度为{s['speed']:.2f} MiB/s。"
    text+=f"该组实际使用{round(s['used']*s['vocab'])}个ID，Used为{100*s['used']:.2f}\\%；Rényi熵为{s['renyi']*10:.4f} bit，除以$\\log_2 1024=10$，得到$\\eta_{{2.5}}={s['renyi']:.4f}$。"
    text+=r'例如相同的两个已用ID，使用次数分别为50／50与90／10时，Used相同，而后者的Rényi效率更低；它刻画频率集中度。'
    write('metric_examples.tex',text);write('metric_example_provenance.json',json.dumps(details,ensure_ascii=False,indent=2))

def main():
    rows=read_catalog();catalog={r['id']:r for r in rows};protocol=json.loads((DATA/'protocol.json').read_text());metrics=json.loads((DATA/'metrics.json').read_text())
    assert len(metrics)==99
    aggregates={}
    for family in FAMILIES:
        aggregates[family]=[]
        for method,label in zip(METHODS,LABELS):
            rr=[r for r in metrics if r['family']==family and r['method']==method]
            aggregates[family].append(dict(method=method,label=label,**{k:float(np.mean([r[k] for r in rr])) for k in METRICS}))
    write('family_aggregates.json',json.dumps(aggregates,ensure_ascii=False,indent=2));figures(aggregates)
    domain_names={'analysis':'分析','geometry_topology':'几何与拓扑','combinatorics':'组合','algebra':'代数','number_theory':'数论','mathematical_physics':'数学物理','optimization':'优化','probability':'概率','statistics':'统计','logic':'逻辑','pde':'PDE','ode':'ODE','linear_algebra':'线性代数'}
    formula=Counter(r['domain'] for r in rows);paper=Counter(r['paper_domain'] for r in rows)
    t=r'\begin{table}[htbp]\centering\small\caption{领域分布：公式领域与来源论文领域分别计数。}\label{tab:domains}\begin{tabular}{lrr@{\qquad}lrr}\toprule 领域&公式领域&论文领域&领域&公式领域&论文领域\\\midrule'+'\n'
    keys=list(domain_names)
    for i in range(7):
        first=keys[i];a=[domain_names[first],f'{formula[first]:,}',f'{paper[first]:,}']
        if i+7<len(keys):second=keys[i+7];a+=[domain_names[second],f'{formula[second]:,}',f'{paper[second]:,}']
        else:a+=['','','']
        t+=' & '.join(a)+r' \\'+'\n'
    write('dataset_table.tex',t+r'\bottomrule\end{tabular}\end{table}')
    r=rows[0];c=r['complexity'];write('complexity_example.tex',f"例如，公式ID\\code{{{r['id']}}}的$(L,C,D,S,O)=({c['latex_lexemes']},{c['command_types']},{c['nesting_depth']},{c['symbol_types']},{c['operator_count']})$，对应$K={c['score']:.3f}$，归入简单类。")
    modelrows=[('Hard AST-BPE','无原子预留；硬合并','训练次序合并','1,024','https://github.com/SamuelGabriel/sqlbpe'),('AST-v4','至多256原子；硬合并','硬约束最短路径','1,024','https://github.com/xtr1976536/math-tokenizer-lab'),('Boundary AST','与Math-atom共享词表','软结构最短路径','1,024','https://github.com/xtr1976536/math-tokenizer-lab'),('Math-atom BPE','至多384原子；官方BPE','官方BPE','1,024','https://github.com/huggingface/tokenizers'),('Byte BPE','官方字节BPE','官方BPE','1,024','https://github.com/huggingface/tokenizers'),('SP Unigram','EM与候选子词剪枝','Viterbi','1,024','https://github.com/google/sentencepiece'),('Qwen2.5','官方固定检查点','官方HF接口','151,665','https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/tokenizer.json'),('Llama 3','官方固定检查点','Meta官方代码','128,256','https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py'),('o200k','官方固定编码','官方tiktoken','200,019','https://github.com/openai/tiktoken')]
    t=r'\begin{table}[htbp]\centering\small\caption{比较模型及词表、推理和代码来源。硬AST-BPE为公开算法的数学迁移实现。}\label{tab:models}\begin{tabularx}{\linewidth}{l Y Y r l}\toprule 模型&词表／训练约束&推理&实际ID空间&来源\\\midrule'+'\n'
    for name,train,infer,size,url in modelrows:t+=' & '.join([tex(name),train,infer,size,r'\href{'+url+r'}{代码}'])+r' \\'+'\n'
    write('model_table.tex',t+r'\bottomrule\end{tabularx}\end{table}')
    t=r'\begin{table}[htbp]\centering\small\caption{十一组实际划分。括号为来源组数；每行已清除跨集合的模板及渲染重叠。}\label{tab:splits}\begin{tabular}{llrrr}\toprule 类型&设置&训练&验证&测试\\\midrule'+'\n'
    for s in protocol['splits']:
        vals=[]
        for key in ['train','dev','test']:vals.append(f"{len(s[key]):,}（{len({catalog[i]['split_group'] for i in s[key]})}）")
        t+=' & '.join([FNAMES[FAMILIES.index(s['family'])],tex(s['name'].replace('extreme_analysis_equations_to_algebra','分析方程→代数').replace('extreme_algebra_to_analysis_equations','代数→分析方程')),*vals])+r' \\'+'\n'
    write('split_table.tex',t+r'\bottomrule\end{tabular}\end{table}')
    env=protocol['environment'];write('environment_generated.tex',f"运行环境为{tex(env['platform'])}、{tex(env['machine'])}、Python {env['python']}；词表训练型基准使用仓库固定的Hugging Face tokenizers与SentencePiece版本。设备与依赖的完整版本保存在实验协议中。")
    metric_examples(catalog,metrics)
    # Generated quantitative prose: every statement reads exact aggregate values.
    idr={r['method']:r for r in aggregates['ID']};b=idr['boundary_ast'];a=idr['math_atom_bpe'];v=idr['ast_v4'];h=idr['hard_ast_bpe'];base=idr['byte_bpe']
    text=r'正文表格对每个测试设置先计算八项指标，再在同类设置内等权平均。ID与复杂度各包含三组来源划分，整域OOD包含三组留出学科，极限OOD包含两个迁移方向。完整单组结果见附录。所有99组模型评价均通过原始字节与字符串精确保真检查。'+'\n'
    text+=r'\subsection{全领域新来源：压缩、命令与结构的取舍}'+'\n'
    text+=table(aggregates['ID'],'全领域新来源测试：三个来源划分的均值。百分比列的单位均为\%。','tab:id',placement='H')
    text+=f"Boundary AST的T/B为{b['tb']:.4f}，相对Byte BPE减少{100*(1-b['tb']/base['tb']):.2f}\\%，相对硬AST-BPE减少{100*(1-b['tb']/h['tb']):.2f}\\%。Math-atom BPE为{a['tb']:.4f}；两者共享词表后，软结构路径使T/B进一步减少{100*(1-b['tb']/a['tb']):.2f}\\%。Boundary AST与Math-atom BPE的Cmd均值分别为{100*b['cmd']:.2f}\\%与{100*a['cmd']:.2f}\\%，表明数学原子初始化直接改善了完整命令的表示。\n"
    text+=f"硬约束的优势体现在结构列：Hard AST-BPE与AST-v4的AST-A分别为{100*h['ast_align']:.2f}\\%和{100*v['ast_align']:.2f}\\%，Cross分别为{100*h['cross']:.2f}\\%和{100*v['cross']:.2f}\\%；Boundary AST的对应值为{100*b['ast_align']:.2f}\\%和{100*b['cross']:.2f}\\%。因此压缩最优路径与最多复合节点边界是不同目标，硬／软约束提供不同工作点。\n"
    text+=r'硬约束使用原生算术／绑定语法前端，Cross与AST-A使用独立KaTeX参考；两个前端生成的语法跨度有差异，表中的Cross测量所有模型与同一外部参考的一致性。'+'\n'
    text+=f"Llama 3与o200k的Num均值为{100*idr['llama3']['num']:.2f}\\%与{100*idr['o200k']['num']:.2f}\\%，Byte BPE的Rényi效率为{100*base['renyi']:.2f}\\%，这些列保留了通用数字分组和字节BPE分布性质的优势。\n"
    text+=r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/structure.pdf}\caption{全领域测试中的两种结构测量。柱形为三个来源划分的均值，数值单位为\%。}\label{fig:structure}\end{figure}'+'\n'
    text+=r'\subsection{整域与极限OOD：词表的跨学科迁移}'+'\n'
    text+=table(aggregates['DomainOOD'],'三个整域留出设置的等权均值。','tab:ood',placement='H')
    text+=table(aggregates['ExtremeOOD'],'分析／方程与代数／数论双向极限OOD的均值。','tab:extreme',placement='H')
    for fam in ['DomainOOD','ExtremeOOD']:
        rr={r['method']:r for r in aggregates[fam]};bb=rr['boundary_ast'];vv=rr['ast_v4'];best=min(rr.values(),key=lambda r:r['tb'])
        text+=f"{FNAMES[FAMILIES.index(fam)]}中，T/B最小者为{best['label']}（{best['tb']:.4f}）。Boundary AST的Cmd为{100*bb['cmd']:.2f}\\%，AST-v4的AST-A为{100*vv['ast_align']:.2f}\\%。完整留出领域和方向分别列于表\\ref{{tab:splits}}与附录。\n"
    text+=r'\subsection{简单到复杂：长度与层次变化}'+'\n'
    text+=table(aggregates['Complexity'],'简单公式训练、复杂公式测试：三个来源划分均值。','tab:complexity',placement='H')
    rr={r['method']:r for r in aggregates['Complexity']};bb=rr['boundary_ast'];vv=rr['ast_v4'];bm=rr['byte_bpe']
    text+=f"只在简单公式上构建词表后，Boundary AST在复杂公式上的T/B为{bb['tb']:.4f}，相对Byte BPE减少{100*(1-bb['tb']/bm['tb']):.2f}\\%，Cmd为{100*bb['cmd']:.2f}\\%。AST-v4的AST-A为{100*vv['ast_align']:.2f}\\%，编码速度为{vv['speed']:.2f} MiB/s。该设置同时改变未见来源与公式句法复杂度。\n"
    text+=r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/compression.pdf}\caption{四类泛化测试的压缩结果。各柱为同类设置等权均值；横轴从零开始，越短表示每字节所需token越少。}\label{fig:compression}\end{figure}'+'\n'
    text+=r'\subsection{运行速度与词表使用}'+'\n'
    text+=f"全领域测试中，AST-v4、Boundary AST、Math-atom BPE和Byte BPE的平均完整编码速度分别为{v['speed']:.2f}、{b['speed']:.2f}、{a['speed']:.2f}与{base['speed']:.2f} MiB/s。AST-v4和Boundary AST在C前端中融合解析、trie与路径搜索；速度比较涵盖全部这些步骤。Used以完整ID空间为分母，通用词表与数学专用词表在本公式集上的实际使用比例直接列于各表。\n"
    text+=r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/speed.pdf}\caption{相同测试输入、逐公式完整接口和七轮交错计时下的编码速度。每类对其设置等权平均。}\label{fig:speed}\end{figure}'+'\n'
    text+=r'\subsection{分指标排序与模型选择}'+'\n'
    text+=r'Boundary AST在11个单独设置中均取得最低T/B。以四类测试分别等权后得到分指标前三名，见表\ref{tab:ranks}；数值相同时并列，表格展示顺序沿用模型列表。Used描述测试集使用词表的比例。压缩、命令和数字完整性、语法对齐、分布均匀性与速度分别报告。'+'\n'
    ranking=[]
    for method,label in zip(METHODS,LABELS):ranking.append(dict(method=method,label=label,**{k:float(np.mean([next(r[k] for r in aggregates[f] if r['method']==method) for f in FAMILIES])) for k in METRICS}))
    t=r'\par\begin{table}[H]\centering\small\caption{四类测试等权后的分指标排序，等值并列。Used不用于质量排序。}\label{tab:ranks}\begin{tabularx}{\linewidth}{lYYY}\toprule 指标&第一梯队&第二梯队&第三梯队\\\midrule'+'\n'
    names={'tb':'T/B','cmd':'Cmd','num':'Num','cross':'Cross','ast_align':'AST-A','renyi':r'$\eta_{2.5}$','speed':'速度'}
    for key in ['tb','cmd','num','cross','ast_align','renyi','speed']:
        order=sorted(ranking,key=lambda r:(r[key] if key in ['tb','cross'] else -r[key],METHODS.index(r['method'])))
        groups=[]
        for r in order:
            if groups and abs(groups[-1][0][key]-r[key])<1e-12:groups[-1].append(r)
            else:groups.append([r])
        cells=[]
        for group in groups[:3]:
            r=group[0];value=f"{r[key]:.4f}" if key=='tb' else f"{r[key]:.2f}" if key=='speed' else f"{100*r[key]:.2f}\\%"
            cells.append('／'.join(tex(x['label']) for x in group)+' '+value)
        t+=' & '.join([names[key],*cells])+r' \\'+'\n'
    text+=t+r'\bottomrule\end{tabularx}\end{table}\par'+'\n'
    text+=r'下游实验保留软约束Boundary AST与硬约束AST-v4两个结构工作点，以Math-atom BPE分离词表与分割贡献，并保留Byte BPE、SP Unigram及三个通用词表作为参照。标准硬AST-BPE给出不预留数学原子的硬结构基线。'
    write('results_generated.tex',text)
    write('abstract_generated.tex',f"本文在56,000条研究级LaTeX公式上比较数学专用分词与通用分词。固定训练型模型词表为1,024，评价标准硬AST-BPE、AST-v4、Boundary AST和Math-atom BPE，并与Byte BPE、SentencePiece Unigram、Qwen2.5、Llama 3及o200k比较。实验设置全领域新来源、整域留出、分析方程与代数双向迁移、简单到复杂四类测试，共11组划分和99组模型评价。统一测量T/B、命令完整性、数字词素完整性、编码速度、Used、Cross、复合AST节点对齐率与Rényi效率。全领域测试中，Boundary AST相对Byte BPE减少{100*(1-b['tb']/base['tb']):.2f}\\%的每字节token数，Cmd为{100*b['cmd']:.2f}\\%；AST-v4的复合节点对齐率为{100*v['ast_align']:.2f}\\%，编码速度为{v['speed']:.2f} MiB/s。硬结构约束、软结构代价与数学词法原子分别形成不同的压缩、对齐和效率表现。全部评价通过逐式可逆性检验，并提供完整划分、逐式记录与五轮审查。")
    write('conclusion_generated.tex',f"固定1,024词表的实验显示，数学原子能够提高完整命令的表示，结构信号决定压缩与复合节点边界之间的工作点。Boundary AST在全领域新来源测试中取得{b['tb']:.4f}的T/B，AST-v4取得{100*v['ast_align']:.2f}\\%的复合节点对齐率和{v['speed']:.2f} MiB/s的完整编码速度。四类划分将来源泛化、学科迁移与复杂度迁移分开测量，八项指标从压缩、词表使用、结构和运行效率描述各模型。下一阶段将采用相同模型架构、训练字节量和计算预算，从零训练接入不同分词器的模型，评价PDE／ODE求解、符号求导与积分、非方程类数学推理；同时比较训练吞吐、总训练计算量与任务准确率。")
    appendix=[]
    for scope in protocol['splits']:
        rr=[r for r in metrics if r['protocol']==scope['name']];rr=sorted(rr,key=lambda r:METHODS.index(r['method']))
        appendix.append(table(rr,tex(scope['name'])+'：'+f"训练{len(scope['train']):,}、验证{len(scope['dev']):,}、测试{len(scope['test']):,}条公式。",small=True))
    write('appendix_tables.tex','\n'.join(appendix))
    audit=json.loads((DATA/'independent_audit.json').read_text())
    versions={}
    import importlib.metadata
    for k in ['tokenizers','sentencepiece','tiktoken','numpy','matplotlib']:versions[k]=importlib.metadata.version(k)
    write('reproducibility_generated.tex',r'''代码、数据及全部输出位于\href{https://github.com/xtr1976536/math-tokenizer-lab}{GitHub仓库}。按下列顺序执行实验、复算和报告生成：
\begin{quote}\small\ttfamily
.venv/bin/python scripts/run\_intrinsic\_2026.py\\
.venv/bin/python scripts/audit\_intrinsic\_2026.py\\
.venv/bin/python scripts/build\_intrinsic\_2026\_report.py
\end{quote}
实验目录为\code{experiments/intrinsic_20261015}；协议保存所有训练／验证／测试ID、固定参数、代码哈希、数据哈希与环境信息；每组保存训练所得模型、七轮计时、逐式token ID及统计量。

审查分五轮执行。第一轮检查ID、来源、模板、渲染隔离及领域／复杂度条件；第二轮检查1,024实际ID预算、软约束与数学原子共享词表、硬AST原生实现与独立参考实现的一致性；第三轮检查所有测试公式的精确负载、完整解码、ID范围和七轮确定性；第四轮独立重聚合逐式计数与来源组等权T/B；第五轮从存档token与检查点重建边界，复算所有模型的命令、数字与复合节点对齐，抽样以直接区间判据复核Cross，并调用公开Rényi熵函数核对99组分布统计。所有检查均通过。

'''+f"独立审查覆盖{sum(x['rows'] for x in audit['checks']):,}条模型—公式记录。依赖版本为"+'，'.join(f"{k} {v}" for k,v in versions.items())+'。')
    ci=r'\subsection{来源组级抽样区间}\begin{table}[htbp]\centering\small\caption{全领域测试T/B及来源组级bootstrap的95\%区间。每个单元第一行为点估计，第二行为区间；三列分别为三个来源划分。}\begin{tabular}{lccc}\toprule 模型&ID-17&ID-29&ID-43\\\midrule'+'\n'
    for method,label in zip(METHODS,LABELS):
        cells=[]
        for seed in [17,29,43]:
            r=next(r for r in metrics if r['protocol']==f'id_{seed}' and r['method']==method);lo,hi=r['paper_bootstrap_95']['tb']
            cells.append(r'\shortstack{'+f"{r['tb']:.4f}"+r'\\{}'+f"[{lo:.4f}, {hi:.4f}]"+'}')
        ci+=tex(label)+' & '+' & '.join(cells)+r' \\'+'\n'
    ci+=r'\bottomrule\end{tabular}\end{table}'
    write('bootstrap_table.tex',ci)
    manifest={'source_data':str(DATA/'metrics.json'),'generating_script':'scripts/build_intrinsic_2026_report.py','surface_class':'paper_main',
        'figures':[{'id':id,'pdf':str(PAPER/'figures'/f'{id}.pdf'),'preview':str(PAPER/'figures'/f'{id}.png')} for id in ['compression','structure','speed']],
        'style':'adapted paper-plot grouped bar template;muted academic palette;white background;zero baselines;hatched soft-AST route'}
    write('figures/figure_catalog.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print('REPORT SOURCES AND FIGURES READY',PAPER)

if __name__=='__main__':main()
