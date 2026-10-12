"""Restrained final comparison, fixed raw inputs and matched ID budget."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from math_tokenizer.corpus import ROOT

LABELS={'ast_v4':'AST-v4','v5_structural':'v5 RL','hf_bpe':'HF BPE','sentencepiece_bpe':'SP BPE','sentencepiece_unigram':'SP Unigram','wordpiece_bytes':'WordPiece','pathpiece_bpe_vocab':'PathPiece'}
MARKERS={'ast_v4':'*','v5_structural':'D','hf_bpe':'o','sentencepiece_bpe':'s','sentencepiece_unigram':'v','wordpiece_bytes':'P','pathpiece_bpe_vocab':'^'}

def main():
    plt.style.use(ROOT/'styles/academic.mplstyle');metrics=json.loads((ROOT/'results/v6/full/metrics.json').read_text());folder=ROOT/'results/v6/figures';folder.mkdir(exist_ok=True);catalog=[]
    for capacity in (1024,2048):
        fig,axes=plt.subplots(1,2,figsize=(8.4,3.7),layout='constrained')
        for ax,scenario,suite,title in zip(axes,['broad','specialized'],['fresh_test','extreme_domain_test'],['New papers','Category-excluded transfer']):
            rows=[m for m in metrics if m['capacity']==capacity and m['scenario']==scenario and m['suite']==suite and m['group']=='matched_primary']
            for m in rows:
                name=m['method'];color='#567C78' if name=='ast_v4' else '#B48776' if name=='v5_structural' else '#9B9890'
                ax.scatter(m['document_macro_tokens_per_byte'],m['encode_mib_s'],marker=MARKERS.get(name,'o'),s=125 if name=='ast_v4' else 48,color=color,edgecolor='white',linewidth=.5,zorder=3)
                ax.annotate(LABELS.get(name,name),(m['document_macro_tokens_per_byte'],m['encode_mib_s']),xytext=(4,5),textcoords='offset points',fontsize=8,color='#374151')
            ax.set_title(title+' | '+str(capacity)+' IDs');ax.set_xlabel('Paper-macro tokens / UTF-8 byte');ax.set_ylabel('Encoding throughput (MiB/s)');ax.set_yscale('log');ax.margins(x=.17,y=.20)
            ax.set_yticks([1,3,10,30]);ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.text(.02,.02,'Lower x and higher y are preferable',transform=ax.transAxes,fontsize=7.5,color='#6B7280')
        for extension in ('png','svg'):fig.savefig(folder/f'compression_speed_{capacity}.{extension}',dpi=250)
        svg=folder/f'compression_speed_{capacity}.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
        catalog.append({'source_data':'results/v6/full/metrics.json','script':'scripts/plot_v6.py','exports':[f'results/v6/figures/compression_speed_{capacity}.png',f'results/v6/figures/compression_speed_{capacity}.svg'],'surface_class':'internal_review','claim':'Compression versus encoding speed under matched training and token ID budget','self_review':'pending actual PNG inspection'})
    (folder/'figure_catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')

if __name__=='__main__':main()
