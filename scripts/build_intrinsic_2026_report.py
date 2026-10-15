"""Build the revised ICML preprint from frozen results: eight models, six metrics."""
from collections import Counter
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog
from math_tokenizer.ast_boundary_v8 import BoundaryAST
from math_tokenizer.hard_ast_2026 import HardASTBPE
from math_tokenizer.final_models import load_final
from math_tokenizer.metrics_v2 import token_bytes
from math_tokenizer.intrinsic_2026 import per_formula

DATA=ROOT/'experiments/intrinsic_20261015'
PAPER=ROOT/'paper/intrinsic_20261015'
METHODS=['hard_ast_bpe','boundary_ast','math_atom_bpe','byte_bpe','sp_unigram','qwen25','llama3','o200k']
LABELS=['AST-BPE','软AST-BPE','数学BPE','BPE','Unigram','Qwen2.5','Llama 3','o200k']
ENGLISH=['AST-BPE','Soft AST-BPE','Math BPE','BPE','Unigram','Qwen2.5','Llama 3','o200k']
KEYS=['tb','cmd','num','cross','used','speed']
FAMILIES=['ID','DomainOOD','ExtremeOOD','Complexity']
FAMILY_NAMES=['领域内','跨领域','极限泛化','简单到复杂']
PANEL_NAMES=['In-domain','Domain transfer','Extreme transfer','Simple to complex']
COLORS=['#555A5E','#197D83','#B16A4C','#A0A5AA','#A79A7C','#8D97B6','#839F92','#AD91AA']
MARKERS=['s','o','D','^','v','P','X','h']

def write(name,text):
    p=PAPER/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.rstrip()+'\n')

def literal(s):
    special={'\\':r'\textbackslash{}','{':r'\{','}':r'\}','$':r'\$','&':r'\&','_':r'\_','%':r'\%','#':r'\#','^':r'\textasciicircum{}','~':r'\textasciitilde{}'}
    return r'\texttt{'+''.join(special.get(c,c) for c in s)+'}'

def result_table(rows,caption,label,wide=True):
    env='table';placement='H'
    style=r'\centering\fontsize{8}{9.4}\selectfont\setlength{\tabcolsep}{2.0pt}\renewcommand{\arraystretch}{1.06}' if wide else r'\centering\small\renewcommand{\arraystretch}{1.05}'
    lines=[r'\begin{'+env+'}['+placement+']',style,r'\caption{'+caption+r'}\label{'+label+'}',
           r'\begin{tabular}{lrrrrrr}\toprule',r'模型&T/B $\dn$&Cmd $\up$&Num $\up$&Cross $\dn$&Used&速度 $\up$\\\midrule']
    best={k:(min if k in ['tb','cross'] else max)(r[k] for r in rows) for k in KEYS if k!='used'}
    for i,r in enumerate(rows):
        if i==5:lines.append(r'\midrule')
        cells=[]
        for k in KEYS:
            v=f"{r[k]:.4f}" if k=='tb' else f"{r[k]:.2f}" if k=='speed' else f"{r[k]*100:.2f}"
            if k in best and abs(r[k]-best[k])<1e-12:v=r'\textbf{'+v+'}'
            cells.append(v)
        lines.append(r['label']+' & '+' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule\end{tabular}',r'\end{'+env+'}'])+'\n'

def export(fig,name,claim):
    d=PAPER/'figures';d.mkdir(exist_ok=True)
    fig.savefig(d/f'{name}.pdf',bbox_inches='tight',pad_inches=0.04)
    fig.savefig(d/f'{name}.png',dpi=300,bbox_inches='tight',pad_inches=0.04);plt.close(fig)
    return {'id':name,'claim':claim,'pdf':str(d/f'{name}.pdf'),'preview':str(d/f'{name}.png')}

def figures(metrics,aggregates):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.labelsize':8.5,
        'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'xtick.labelsize':7.5,
        'ytick.labelsize':7.5,'xtick.major.width':0.6,'ytick.major.width':0.6,'pdf.fonttype':42,'ps.fonttype':42,
        'legend.frameon':False,'figure.facecolor':'white','axes.facecolor':'white'})
    catalog=[];fig,axes=plt.subplots(1,4,figsize=(6.7,2.65),sharey=True)
    for ax,family,title,letter in zip(axes,FAMILIES,PANEL_NAMES,'abcd'):
        for i,m in enumerate(METHODS):
            vals=[r['tb'] for r in metrics if r['family']==family and r['method']==m]
            ax.plot([min(vals),max(vals)],[i,i],color=COLORS[i],lw=1.2,zorder=2)
            ax.scatter(vals,np.full(len(vals),i),s=9,color=COLORS[i],alpha=0.38,zorder=3)
            ax.scatter(np.mean(vals),i,s=24 if i in [1,2] else 17,marker=MARKERS[i],color=COLORS[i],edgecolors='white',lw=0.35,zorder=4)
        ax.set_title(f'({letter}) {title}',loc='left',pad=7,fontsize=8.1)
        ax.set_xlim(0.35,0.57);ax.set_xticks([0.36,0.44,0.52]);ax.set_xlabel('Tokens / byte ↓');ax.grid(axis='x',color='#E5E7E9',lw=0.5,zorder=0)
        ax.set_ylim(7.55,-0.55);ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(8),ENGLISH)
    fig.subplots_adjust(left=0.15,right=0.99,bottom=0.20,top=0.86,wspace=0.20)
    catalog.append(export(fig,'compression','Points show split values;line shows range and large marker the equal-setting mean.'))
    fig,axes=plt.subplots(1,2,figsize=(6.7,2.55))
    for i,m in enumerate(METHODS):
        rr=[r for r in metrics if r['family']=='ID' and r['method']==m]
        x=np.array([r['tb'] for r in rr]);cross=np.array([r['cross']*100 for r in rr]);speed=np.array([r['speed'] for r in rr])
        for ax,ys in [(axes[0],cross),(axes[1],speed)]:
            ax.scatter(x,ys,s=10,color=COLORS[i],alpha=0.28,marker=MARKERS[i],zorder=2)
            ax.scatter(x.mean(),ys.mean(),s=44 if i in [0,1,2] else 30,marker=MARKERS[i],color=COLORS[i],edgecolors='white',lw=0.5,zorder=4)
    for ax in axes:
        ax.set_xlabel('Tokens / byte ↓');ax.set_xlim(0.35,0.55);ax.set_xticks([0.36,0.42,0.48,0.54]);ax.grid(color='#E5E7E9',lw=0.5);ax.set_axisbelow(True)
    axes[0].set_ylabel('Crossing tokens (%) ↓');axes[0].set_ylim(0,29);axes[0].set_title('(a) Compression and crossing',loc='left')
    axes[1].set_ylabel('Encoding speed (MiB/s) ↑');axes[1].set_ylim(0,34);axes[1].set_title('(b) Compression and speed',loc='left')
    legend=[Line2D([0],[0],marker=MARKERS[i],linestyle='',color=COLORS[i],label=ENGLISH[i],markersize=5) for i in range(8)]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(0.53,1.15),ncol=4,columnspacing=1.6,handletextpad=0.3,fontsize=7.7)
    fig.subplots_adjust(left=0.08,right=0.99,bottom=0.20,top=0.84,wspace=0.32)
    catalog.append(export(fig,'tradeoffs','In-domain compression, crossing and speed;each of three splits is shown.'))
    write('figures/figure_catalog.json',json.dumps({'source':str(DATA/'metrics.json'),'generator':'scripts/build_intrinsic_2026_report.py','surface_class':'paper_main',
        'style':'ICML-sized vector point/range plots;split observations;consistent colour/marker;no truncated bars','figures':catalog},ensure_ascii=False,indent=2))

def examples(catalog,metrics):
    f=DATA/'id_17/models';records=[catalog['0e5a942aa5faafdca707467a'],catalog['06a371ce545de9ef798775c5']]
    models={'AST-BPE':HardASTBPE.load(f/'hard_ast_bpe.json'),'软AST-BPE':BoundaryAST.load(f/'boundary_ast.json'),'Qwen2.5':load_final('Qwen2.5-0.5B')}
    lines=[r'\begin{table}[H]\centering\footnotesize',r'\caption{测试公式的实际分词。竖线表示token边界。}\label{tab:example}',r'\begin{tabularx}{\columnwidth}{lY}\toprule 模型&token序列\\\midrule'];info={}
    for letter,row in zip(['A','B'],records):
        lines.append(r'\multicolumn{2}{l}{'+letter+'：$'+row['latex']+'$}'+r'\\[2pt]');info[letter]={}
        for label,m in models.items():
            ids=m.encode(row['latex']);parts=[token_bytes(m,i).decode() for i in ids];counts=per_formula(m,row,ids)
            info[letter][label]={'pieces':parts,'counts':counts}
            seq=r' $|$\allowbreak '.join(literal(x) for x in parts)
            lines.append(label+' & '+seq+r'\\')
        if letter=='A':lines.append(r'\midrule')
    write('example_table.tex','\n'.join(lines+[r'\bottomrule\end{tabularx}',r'\end{table}']))
    s=next(r for r in metrics if r['protocol']=='id_17' and r['method']=='boundary_ast');a=info['A'];b=info['B']
    text=r'''\paragraph{T/B $\dn$：压缩。}
单式T/B等于token数除以原串字节数。表\ref{tab:example}的公式A有11字节，软AST-BPE与AST-BPE均分成5个token：
\[\mathrm{T/B}_A=5/11\approx0.4545.\]
Qwen2.5使用7个token，对应$7/11\approx0.6364$。实验表先计算每个来源组的总token数／总字节数，再对来源组取平均：
\[\mathrm{T/B}=\frac{1}{G}\sum_{g=1}^G\frac{T_g}{B_g}.\]

\paragraph{Cmd $\up$：命令完整率。}
公式A只有一个命令\code{\frac}。AST-BPE和软AST-BPE均未在命令内部切开，Cmd为$1/1=100\%$；Qwen2.5将反斜杠与\code{frac}分开，Cmd为$0/1=0\%$。汇总时用完整命令数除以命令总数。

\paragraph{Num $\up$：数字完整率。}
公式B含数字1和12。AST-BPE保留两个数字，Num为$2/2=100\%$；软AST-BPE在12内部切开，Num为$1/2=50\%$。汇总时用内部未切开的数字词素数除以数字词素总数。

\paragraph{Cross $\dn$：语法部分跨越率。}
按KaTeX语法跨度判断token是否部分越过节点。公式A的软AST-BPE将\code{\frac}与左花括号合并，5个token中有1个跨越分子组起点：
\[\mathrm{Cross}_A=1/5\times100\%=20\%.\]
AST-BPE保留分子组，Cross为$0/5=0\%$。实验表用跨越token数除以全部token数，每个token最多计一次。

\paragraph{Used：词表使用率。}
统计输入中出现过的不同ID，除以词表大小：
\[\mathrm{Used}=\frac{\text{已使用的不同ID数}}{\text{词表ID总数}}\times100\%.\]
'''
    unique=len(set(b['软AST-BPE']['counts']['ids']))
    text+=f"公式B在软AST-BPE下使用{unique}个不同ID，因此单式Used为${unique}/1024\\times100\\%={100*unique/1024:.4f}\\%$。该组完整测试集使用{round(s['used']*s['vocab'])}个ID，Used为${round(s['used']*s['vocab'])}/1024\\times100\\%={100*s['used']:.2f}\\%$。\n"
    text+=r'''\paragraph{速度 $\up$：完整编码吞吐。}
用原始输入MiB除以完整编码时间：
\[\mathrm{Speed}=\frac{\text{输入字节数}/2^{20}}{\text{编码秒数}}.\]
'''
    seconds=float(np.median(s['timing_seconds']));text+=f"同组公式A、B所在的完整测试集有{s['bytes']:,}字节；软AST-BPE七轮编码的中位耗时为{seconds:.8f}秒：\n"
    text+=r'\['+f"\\mathrm{{Speed}}=\\frac{{{s['bytes']}/2^{{20}}}}{{{seconds:.8f}}}\\approx{s['speed']:.2f}\\ \mathrm{{MiB/s}}."+r'\]'+'\n'
    text+='计时包含原子映射、解析和分割，模型加载与指标计算在计时之外。\n'
    write('metric_definitions.tex',text);write('metric_example_provenance.json',json.dumps(info,ensure_ascii=False,indent=2))
    for letter,row in zip(['A','B'],records):
        for label in models:assert ''.join(info[letter][label]['pieces'])==row['latex']
    assert a['软AST-BPE']['counts']['crossings']==1 and a['AST-BPE']['counts']['crossings']==0
    assert b['AST-BPE']['counts']['number_good']==2 and b['软AST-BPE']['counts']['number_good']==1

def main():
    metrics=[r for r in json.loads((DATA/'metrics.json').read_text()) if r['method'] in METHODS];assert len(metrics)==88
    rows=read_catalog();catalog={r['id']:r for r in rows};protocol=json.loads((DATA/'protocol.json').read_text())
    for r in metrics:r['label']=LABELS[METHODS.index(r['method'])]
    aggregates={f:[dict(method=m,label=LABELS[i],**{k:float(np.mean([r[k] for r in metrics if r['family']==f and r['method']==m])) for k in KEYS}) for i,m in enumerate(METHODS)] for f in FAMILIES}
    write('family_aggregates.json',json.dumps(aggregates,ensure_ascii=False,indent=2));figures(metrics,aggregates);examples(catalog,metrics)
    write('sources_generated.tex',r'''\begin{table}[H]\centering\small
\caption{Benchmark的公式来源与数量。}\label{tab:sources}
\begin{tabularx}{\columnwidth}{Yr}\toprule 来源&公式数\\\midrule
\href{https://arxiv.org/}{2020年前arXiv论文}&51,229\\
\href{https://github.com/openai/math}{OpenAI数学仓库}&2,810\\
\href{https://github.com/fredrik-johansson/fungrim}{FunGrim}&1,889\\
\href{https://github.com/sympy/sympy}{SymPy}&72\\\midrule
合计&56,000\\\bottomrule\end{tabularx}\end{table}''')
    formula=Counter(r['domain'] for r in rows);names={'analysis':'分析','geometry_topology':'几何拓扑','combinatorics':'组合','algebra':'代数','number_theory':'数论','mathematical_physics':'数学物理','optimization':'优化','probability':'概率','statistics':'统计','logic':'逻辑','pde':'PDE','ode':'ODE','linear_algebra':'线性代数'}
    t=r'\begin{table}[H]\centering\small\caption{按公式内容划分的领域分布。}\label{tab:domains}\begin{tabular}{lr@{\quad}lr}\toprule 领域&数量&领域&数量\\\midrule'+'\n';keys=list(names)
    for i in range(7):
        a=keys[i];cells=[names[a],f'{formula[a]:,}']
        if i+7<len(keys):b=keys[i+7];cells.extend([names[b],f'{formula[b]:,}'])
        else:cells.extend(['',''])
        t+=' & '.join(cells)+r'\\'+'\n'
    write('dataset_table.tex',t+r'\bottomrule\end{tabular}\end{table}')
    c=catalog['00fb843edd55868789be31dc']['complexity']
    assert (c['latex_lexemes'],c['command_types'],c['nesting_depth'],c['symbol_types'],c['operator_count'])==(11,1,2,5,2)
    write('complexity_example.tex',r'''例如，语料中的公式$s^{\frac{1}{3}}$：
\begin{itemize}
\item 源码为\texttt{s\textasciicircum{}\{\textbackslash{}frac\{1\}\{3\}\}}，词素数$L=11$。
\item 只有\code{\frac}一种命令，$C=1$；括号嵌套深度$D=2$。
\item 非括号符号为$s$、\code{^}、\code{\frac}、1、3，故$S=5$；运算符为\code{^}和\code{\frac}，$O=2$。
\end{itemize}
代入式\eqref{eq:complexity}：
\[
\begin{aligned}
K={}&0.40\frac{11}{60}+0.20\frac{1}{6}+0.15\frac{2}{3}\\
&+0.15\frac{5}{18}+0.10\frac{2}{12}\\
={}&0.265<0.65,
\end{aligned}
\]
因此该公式属于简单类。''')
    links=['https://github.com/SamuelGabriel/sqlbpe','https://github.com/xtr1976536/math-tokenizer-lab','https://github.com/huggingface/tokenizers','https://github.com/huggingface/tokenizers','https://github.com/google/sentencepiece','https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/tokenizer.json','https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py','https://github.com/openai/tiktoken']
    t=r'\begin{table}[t]\centering\small\caption{模型容量与代码来源。AST-BPE采用公开算法的数学迁移实现。}\label{tab:models}\begin{tabular}{lrr}\toprule 模型&ID空间&来源\\\midrule'+'\n'
    for i,m in enumerate(METHODS):
        n=next(r['vocab'] for r in metrics if r['method']==m);t+=LABELS[i]+f' & {n:,} & '+r'\href{'+links[i]+r'}{代码}\\'+'\n'
    write('model_table.tex',t+r'\bottomrule\end{tabular}\end{table}')
    display={}
    for f in FAMILIES:
        for i,s in enumerate([s for s in protocol['splits'] if s['family']==f]):
            label=f'划分{i+1}' if f in ['ID','Complexity'] else ['几何／组合／数论','概率／统计／优化','分析／PDE／ODE'][i] if f=='DomainOOD' else ['分析方程→代数','代数→分析方程'][i]
            display[s['name']]=(FAMILY_NAMES[FAMILIES.index(f)],label)
    t=r'\begin{table*}[t]\centering\small\caption{四类实验的实际公式数量。跨领域行名表示测试领域；极限泛化行名表示迁移方向。}\label{tab:splits}\begin{tabular}{llrrr}\toprule 实验&划分／测试领域&训练&验证&测试\\\midrule'+'\n'
    for f in FAMILIES:
        for s in protocol['splits']:
            if s['family']==f:t+=' & '.join([*display[s['name']],*[f'{len(s[k]):,}' for k in ['train','dev','test']]])+r'\\'+'\n'
        if f!=FAMILIES[-1]:t+=r'\addlinespace'+'\n'
    write('split_table.tex',t+r'\bottomrule\end{tabular}\end{table*}')
    captions=['领域内实验：三个来源划分的均值。','跨领域实验：三个留出组合的均值。','极限泛化实验：两个迁移方向的均值。','简单到复杂实验：三个来源划分的均值。'];labels=['tab:id','tab:ood','tab:extreme','tab:complexity']
    write('core_results.tex','\n'.join(result_table(aggregates[f],c,l) for f,c,l in zip(FAMILIES,captions,labels)))
    idr={r['method']:r for r in aggregates['ID']};soft=idr['boundary_ast'];mathb=idr['math_atom_bpe'];hard=idr['hard_ast_bpe'];bpe=idr['byte_bpe']
    gains=[100*(1-next(r['tb'] for r in aggregates[f] if r['method']=='boundary_ast')/next(r['tb'] for r in aggregates[f] if r['method']=='byte_bpe')) for f in FAMILIES]
    analysis=r'\begin{itemize}'+'\n'
    analysis+=f"\\item \\textbf{{压缩：}}软AST-BPE在11组划分中均取得最低T/B；四类实验相对BPE分别减少{gains[0]:.2f}\\%、{gains[1]:.2f}\\%、{gains[2]:.2f}\\%和{gains[3]:.2f}\\%。\n"
    analysis+=f"\\item \\textbf{{命令与数字：}}领域内实验中，软AST-BPE和数学BPE的Cmd均为{100*soft['cmd']:.2f}\\%，BPE为{100*bpe['cmd']:.2f}\\%；Llama 3和o200k的Num均为{100*idr['llama3']['num']:.2f}\\%，高于数学专用模型。\n"
    analysis+=f"\\item \\textbf{{结构与速度：}}AST-BPE的Cross为{100*hard['cross']:.2f}\\%，软AST-BPE为{100*soft['cross']:.2f}\\%；硬约束产生更少部分跨越。软AST-BPE速度为{soft['speed']:.2f} MiB/s，数学BPE为{mathb['speed']:.2f} MiB/s。\n"
    write('analysis_generated.tex',analysis+r'\end{itemize}')
    write('result_figures.tex',r'''\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/compression.pdf}
\caption{四类实验的T/B。小点为各次来源划分或迁移设置，线段为范围，大标记为等权均值；每列分别包含3、3、2、3组设置。}\label{fig:compression}
\end{figure*}
\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/tradeoffs.pdf}
\caption{领域内三个来源划分的压缩、Cross与编码速度。小点为单次划分，大标记为均值。左图比较压缩与语法跨越，右图比较完整编码接口吞吐。}\label{fig:tradeoffs}
\end{figure*}''')
    write('abstract_generated.tex',f"本文构建包含56,000条LaTeX公式的数学分词benchmark，按领域、来源和复杂度分类。我们研究数学原子BPE与软AST-BPE，并与标准AST-BPE及五个基准比较。实验包含领域内、跨领域、极限泛化和简单到复杂四类设置，使用六项指标评价压缩、数学单位完整性、语法跨越、词表使用与编码速度。软AST-BPE在11组划分中均取得最低T/B，领域内相对BPE减少{gains[0]:.2f}\\%的每字节token数，命令完整率为{100*soft['cmd']:.2f}\\%。标准AST-BPE的Cross更低，软AST-BPE则获得更好的压缩与编码吞吐。")
    write('conclusion_generated.tex',f"本文构建了具有领域、来源和复杂度分类的数学公式benchmark，并用四类泛化实验比较八个分词器。数学BPE通过完整命令原子改善数学单位表示；软AST-BPE在共享词表上加入结构代价，在11组划分中均获得最低T/B，领域内命令完整率为{100*soft['cmd']:.2f}\\%，编码速度为{soft['speed']:.2f} MiB/s。标准AST-BPE取得更低Cross，显示硬约束与软约束各自的结构与压缩优势。")
    appendix=[]
    for f in FAMILIES:
        for s in protocol['splits']:
            if s['family']!=f:continue
            rr=sorted([r for r in metrics if r['protocol']==s['name']],key=lambda r:METHODS.index(r['method']));name,setting=display[s['name']]
            appendix.append(result_table(rr,name+'：'+setting+'。','tab:detail'+str(len(appendix)+1),wide=False))
    write('appendix_tables.tex','\n'.join(appendix))
    review={'methods':METHODS,'labels':LABELS,'metrics':KEYS,'records':len(metrics),'formula_method_records':sum(r['formulas'] for r in metrics),
        'data_sha256':hashlib.sha256((DATA/'metrics.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((DATA/'protocol.json').read_bytes()).hexdigest(),
        'raw_results_unchanged':True,'excluded_methods':['ast_v4'],'excluded_metrics':['ast_align','renyi'],
        'layout':'official ICML2026 preprint;Chinese fonts;two-column main text','checks':{'four_families':len(aggregates)==4,'all_six_metrics':all(set(KEYS)<=set(r) for rr in aggregates.values() for r in rr),'fixed_capacities':all(r['vocab']==1024 for r in metrics if r['method'] in METHODS[:5])}}
    write('revision_validation.json',json.dumps(review,ensure_ascii=False,indent=2))
    print('Revised manuscript ready:',PAPER,'records',len(metrics),'formula-method rows',review['formula_method_records'])

if __name__=='__main__':main()
