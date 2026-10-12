"""Standalone figures from frozen complete results; no selection or fitting."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from math_tokenizer.corpus import ROOT


def main():
    folder=ROOT/'results/v8/full';ranks=json.loads((folder/'rankings.json').read_text())
    decisions=json.loads((ROOT/'results/v8/controlled_terminal/decision.json').read_text())
    dest=ROOT/'results/v8/figures';dest.mkdir(exist_ok=True)
    names=['ast_v4_parent','official_atomic_bpe','ast_lexicographic','ast_boundary','v5_parent']
    labels=['Hard AST','Atomic BPE','Length-first AST','Boundary AST','Legacy v5']
    colors=['#0072B2','#666666','#E69F00','#009E73','#CC79A7']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,(capacity,scenario) in zip(axes.flat,[(1024,'broad'),(1024,'specialized'),(2048,'broad'),(2048,'specialized')]):
        metrics=ranks['within_scenario'][f'{capacity}/{scenario}']['metrics']
        x=np.arange(len(names));base=metrics['official_atomic_bpe']['document_macro_tokens_per_byte']
        values=[100*(1-metrics[n]['document_macro_tokens_per_byte']/base) for n in names]
        ax.bar(x,values,color=colors,width=.55,alpha=.85)
        ax.axhline(0,color='#333333',lw=.8);ax.set_xticks(x,labels,rotation=20,ha='right')
        ax.set_ylabel('Sequence reduction vs atomic BPE (%)')
        ax.set_title(f'{capacity} IDs | {scenario}',loc='left',fontweight='bold')
        right=ax.twinx();right.spines['top'].set_visible(False)
        recalls=[100*metrics[n]['independent_scope_boundary_recall'] for n in names]
        right.plot(x,recalls,'o-',color='#222222',lw=1.2,markersize=4)
        right.set_ylim(0,80);right.set_ylabel('Independent scope recall (%)')
        ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.suptitle('Compression and source-boundary preservation on sealed v8 formulas',fontweight='bold')
    for ext in ('png','pdf','svg'):fig.savefig(dest/f'ast_tradeoffs.{ext}',dpi=220)
    plt.close(fig)
    source=json.loads((ROOT/'results/v8/controlled_terminal/metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    for ax,control in zip(axes,('random_search','cem_search')):
        for i,fold in enumerate((211,337,461)):
            values=[]
            for seed in (17,29,43,59,71):
                a=next(m for m in source if m['fold']==fold and m['seed']==seed and m['method']=='parameter_rl')
                b=next(m for m in source if m['fold']==fold and m['seed']==seed and m['method']==control)
                values.append(b['terminal_validation_utility']-a['terminal_validation_utility'])
            ax.scatter(np.arange(5)+(i-1)*.12,values,s=34,color=['#0072B2','#009E73','#E69F00'][i],label=f'Paper split {fold}')
        ax.axhline(0,color='#333333',lw=1);ax.set_xticks(range(5),['17','29','43','59','71'])
        ax.set_ylim(-.011,.011)
        ax.set_xlabel('Training seed');ax.set_ylabel('Utility improvement of RL (higher is better)')
        p=decisions['comparisons'][control];ci=p['seed_cluster_ci95']
        ax.set_title(f"RL vs {control.replace('_',' ')} | wins {p['winning_pairs']}/15",loc='left',fontweight='bold')
        ax.text(.02,.98,f"Mean {p['utility_improvement_mean']:+.5f}\nSeed-cluster 95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]",transform=ax.transAxes,va='top',fontsize=9,
            bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
        ax.grid(axis='y',alpha=.16)
    handles,legend_labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,-.06),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Same multi-objective reward and 34 terminal evaluations per optimizer',fontweight='bold')
    for ext in ('png','pdf','svg'):fig.savefig(dest/f'rl_controlled_seeds.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    for path in dest.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print('Frozen-result figures written',flush=True)


if __name__=='__main__':main()
