"""Exactly three evidence figures, sized for the supplied 4:3 Beamer slides."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.active_benchmark import read_catalog

METHODS=['ast_boundary','official_atomic_bpe','hf_byte_bpe','sentencepiece_unigram','Qwen2.5-0.5B','meta-llama/Meta-Llama-3-8B','tiktoken_o200k_base']
LABELS=['Boundary AST','Math-atom BPE','Byte BPE','SP Unigram','Qwen2.5','Llama 3','o200k']
COLORS=['#0072B2','#009E73','#888888','#CC79A7','#D55E00','#E69F00','#56B4E9']


def save(fig,name):
    folder=ROOT/'presentations/math_tokenizer/source/assets'
    for ext in ('pdf','png'):fig.savefig(folder/(name+'.'+ext),dpi=260,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.linewidth':.7})
    rows=read_catalog();domains={r['domain'] for r in rows};ordered=sorted(domains,key=lambda d:-sum(r['domain']==d for r in rows))
    labels={'geometry_topology':'Geometry / topology','mathematical_physics':'Math. physics','number_theory':'Number theory','linear_algebra':'Linear algebra','pde':'PDE','ode':'ODE'}
    fig,(ax,right)=plt.subplots(1,2,figsize=(10,4.4),gridspec_kw={'width_ratios':[1.8,1]},layout='constrained')
    y=np.arange(len(ordered));simple=[sum(r['domain']==d and r['complexity']['class']=='simple' for r in rows) for d in ordered]
    complex_=[sum(r['domain']==d and r['complexity']['class']=='complex' for r in rows) for d in ordered]
    ax.barh(y,simple,color='#547BA5',label='Simple');ax.barh(y,complex_,left=simple,color='#58A68A',label='Complex')
    ax.set_yticks(y,[labels.get(d,d.replace('_',' ').capitalize()) for d in ordered],fontsize=9);ax.invert_yaxis()
    ax.set_xlabel('Formula count');ax.set_xlim(0,24500);ax.set_title('(a) Domains and syntax complexity',loc='left',fontweight='bold')
    for i,(a,b) in enumerate(zip(simple,complex_)):ax.text(a+b+300,i,f'{a+b:,}',va='center',fontsize=8)
    ax.legend(frameon=False,loc='lower right',fontsize=9);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    origin=['Historical human','Possible mixed','Unknown'];counts=[51229,2810,1961]
    right.barh([2,1,0],counts,color=['#547BA5','#D7B879','#AAAAAA'],height=.5)
    right.set_yticks([2,1,0],origin,fontsize=9);right.set_xlim(0,63500);right.set_xlabel('Formula count')
    for y_,n in zip([2,1,0],counts):right.text(n+1000,y_,f'{n:,}',va='center',fontsize=9)
    right.set_title('(b) Source provenance',loc='left',fontweight='bold');right.grid(axis='x',alpha=.16);right.set_axisbelow(True)
    save(fig,'dataset_composition')
    ranks=json.loads((ROOT/'experiments/current/rankings.json').read_text())['boards']
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax,scenario,title in zip(axes,['broad','specialized'],['(a) New-paper test: 4,294 formulas','(b) Source-domain holdout: 1,428 formulas']):
        for name,label,color in zip(METHODS,LABELS,COLORS):
            values=[ranks[f'{c}/{scenario}']['metrics'][name]['document_macro_tokens_per_byte'] for c in (1024,2048)]
            ax.plot([1024,2048],values,marker='o',markersize=4,color=color,linewidth=1.6,ls='--' if name in METHODS[4:] else '-',label=label)
        ax.set_xticks([1024,2048],['1,024','2,048']);ax.set_xlabel('Trained vocabulary budget (IDs)')
        ax.set_ylabel('Paper-macro tokens / UTF-8 byte (lower is better)');ax.grid(alpha=.18)
        ax.set_title(title,loc='left',fontweight='bold');ax.set_ylim(.30,.62)
    handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',bbox_to_anchor=(.5,-.16),ncol=4,frameon=False,fontsize=9)
    save(fig,'compression_by_budget')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,c in zip(axes,[1024,2048]):
        data=ranks[f'{c}/broad']['metrics']
        label_positions=[(23,15),(6,16.5),(6,28),(19,26),(2.3,23.7),(5.7,22.7),(11.9,24.4)] if c==1024 else [(23,20),(5.2,26),(6.5,33),(19,28),(2.3,23.6),(5.7,19),(13.5,16.5)]
        for n,label,color,position in zip(METHODS,LABELS,COLORS,label_positions):
            m=data[n];x=m['encode_mib_s'];y=100*m['independent_tree_crossing']
            ax.scatter(x,y,s=60,color=color,marker='D' if n in METHODS[:2] else 'o',zorder=3)
            ax.annotate(label,(x,y),xytext=position,textcoords='data',fontsize=9,color='#222222',
                arrowprops={'arrowstyle':'-','lw':.6,'color':'#888888','shrinkB':6})
        ax.set_xscale('log');ax.set_xlim(2,60);ax.set_ylim(10,38);ax.set_xticks([2,5,10,20,40],['2','5','10','20','40'])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Full encode throughput (MiB/s; log scale)')
        ax.set_ylabel('KaTeX partial-crossing tokens (%)')
        ax.set_title(f'({"a" if c==1024 else "b"}) {c:,} IDs: new-paper test',loc='left',fontweight='bold')
        ax.grid(alpha=.16);ax.text(.02,.97,'Lower and farther right is better',transform=ax.transAxes,va='top',fontsize=9,color='#555555')
    save(fig,'speed_structure')
    print('Three presentation figures generated')


if __name__=='__main__':main()
