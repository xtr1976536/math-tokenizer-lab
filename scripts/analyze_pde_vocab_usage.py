"""Analyze actual output-token use across PDE source-isolated folds."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from math_tokenizer.corpus import ROOT

EXPERIMENT = ROOT / 'experiments/pde1500_sweep_200_450_20261022'
OUTPUT = ROOT / 'reports/pde1500_vocab_200_450_20261022'
METHODS = ('ast_bpe', 'soft_ast_bpe')


def pooled_metrics(results):
    grouped = defaultdict(Counter)
    for result in results:
        protocol = 'three_fold' if result['protocol'].startswith('pde_fold_') else 'simple_to_complex'
        grouped[(protocol, result['method'], result['total_ids'])].update(result['counts'])
    rows = []
    for (protocol, method, size), c in grouped.items():
        rows.append(dict(protocol=protocol, method=method, total_ids=size,
            compression_ratio=c['initial_character_tokens']/c['final_tokens'],
            cmd_pct=100*c['commands_intact']/c['commands'],
            num_pct=100*c['numbers_intact']/c['numbers']))
    return sorted(rows, key=lambda r:(r['protocol'], r['method'], r['total_ids']))


def usage_rows():
    rows = []
    for fold in range(1, 4):
        folder = EXPERIMENT / f'pde_fold_{fold}'
        for method in METHODS:
            for path in sorted(folder.glob(f'{method}_*.tokens.csv')):
                size = int(path.stem.split('_')[-1].split('.')[0])
                with path.open(newline='') as f:
                    for row in csv.DictReader(f):
                        row.update(fold=fold, method=method, total_ids=size)
                        for field in ('content_id','bytes','train_count','train_formulas','train_papers',
                                      'test_count','test_formulas','test_papers'):
                            row[field] = int(row[field])
                        for field in ('train_probability','test_probability'):
                            row[field] = float(row[field])
                        rows.append(row)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'], row['total_ids'])].append(row)
    summaries = []
    for (method, size), tokens in sorted(groups.items()):
        learned = [r for r in tokens if r['type']=='learned_merge']
        commands = [r for r in tokens if r['type']=='command_atom']
        train_total = sum(r['train_count'] for r in tokens)
        test_total = sum(r['test_count'] for r in tokens)
        rare = [r for r in learned if r['train_count']<=10]
        single = [r for r in learned if r['train_papers']<=1]
        long_rare = [r for r in learned if r['bytes']>=16 and r['train_count']<=10]
        row = dict(method=method,total_ids=size,folds=3,
            base_character_ids=sum(r['type']=='base_character' for r in tokens),
            command_atom_ids=len(commands),learned_merge_ids=len(learned),
            learned_zero_train_ids=sum(r['train_count']==0 for r in learned),
            learned_zero_test_ids=sum(r['test_count']==0 for r in learned),
            learned_train_le_2_ids=sum(r['train_count']<=2 for r in learned),
            learned_train_le_5_ids=sum(r['train_count']<=5 for r in learned),
            learned_train_le_10_ids=len(rare),
            learned_single_paper_ids=len(single),
            learned_long_rare_ids=len(long_rare),
            max_learned_bytes=max((r['bytes'] for r in learned),default=0),
            learned_train_le_10_token_mass=sum(r['train_count'] for r in rare)/train_total,
            learned_single_paper_token_mass=sum(r['train_count'] for r in single)/train_total,
            learned_test_zero_train_ids=sum(r['train_count']==0 and r['test_count']>0 for r in learned),
            train_token_total=train_total,test_token_total=test_total)
        summaries.append(row)
    return summaries


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot(metrics, summaries, rows):
    colors={'ast_bpe':'#2575a8','soft_ast_bpe':'#c45b34'}
    labels={'ast_bpe':'Hard AST-BPE','soft_ast_bpe':'Soft AST-BPE'}
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for j,(key,title) in enumerate([('compression_ratio','Compression'),('cmd_pct','Cmd (%)'),('num_pct','Num (%)')]):
        ax=axes[0,j]
        for method in METHODS:
            subset=[r for r in metrics if r['protocol']=='three_fold' and r['method']==method]
            ax.plot([r['total_ids'] for r in subset],[r[key] for r in subset],
                    marker='o',ms=3,color=colors[method],label=labels[method])
        ax.set(title=title,xlabel='Total IDs',ylabel=title)
        if key=='num_pct':ax.set_ylim(97,100)
        ax.grid(alpha=.2)
        if j==0:ax.legend()
    for j,(key,title) in enumerate([('learned_zero_train_ids','Unused learned IDs / 3 folds'),
                                    ('learned_single_paper_ids','Learned IDs in <=1 paper / 3 folds'),
                                    ('learned_train_le_10_token_mass','Low-frequency token mass')]):
        ax=axes[1,j]
        for method in METHODS:
            subset=[r for r in summaries if r['method']==method]
            factor=100 if key.endswith('_mass') else 1
            ax.plot([r['total_ids'] for r in subset],[factor*r[key] for r in subset],
                    marker='o',ms=3,color=colors[method],label=labels[method])
        ax.set(title=title,xlabel='Total IDs',ylabel='Percent of train outputs' if key.endswith('_mass') else 'IDs summed across folds')
        ax.grid(alpha=.2)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUTPUT/'size_and_usage.png',dpi=175);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,method in zip(axes,METHODS):
        for size,alpha in [(300,.55),(384,.75),(450,1.0)]:
            subset=sorted((r['train_probability'] for r in rows if r['method']==method and r['total_ids']==size and r['type']=='learned_merge' and r['train_count']>0),reverse=True)
            ax.plot(range(1,len(subset)+1),subset,label=str(size),alpha=alpha)
        ax.set(xlabel='Rank among used learned merges',ylabel='Training output probability',title=labels[method],yscale='log')
        ax.legend(title='Total IDs');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUTPUT/'frequency_ranks.png',dpi=175);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,method in zip(axes,METHODS):
        for size,marker in [(300,'o'),(450,'x')]:
            subset=[r for r in rows if r['fold']==1 and r['method']==method and r['total_ids']==size and r['type']=='learned_merge']
            ax.scatter([r['bytes'] for r in subset],[r['train_count']+.5 for r in subset],
                       s=17,alpha=.45,marker=marker,label=str(size))
        ax.set(xlabel='Token bytes',ylabel='Train output count (+0.5)',title=labels[method],yscale='log')
        ax.legend(title='Total IDs');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUTPUT/'length_vs_frequency.png',dpi=175);plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/benchmark_pde1500_20261022/manifest.json').read_text())
    protocol=json.loads((EXPERIMENT/'protocol.json').read_text())
    audit=json.loads((EXPERIMENT/'audit.json').read_text())
    assert protocol['data_sha256']==manifest['pde_sha256'] and audit['status']=='PASS'
    metrics=pooled_metrics(json.loads((EXPERIMENT/'results.json').read_text()))
    rows=usage_rows();summaries=summarize(rows)
    assert len(metrics)==len(summaries)*2
    for row in summaries:
        assert row['base_character_ids']==98*3
        assert row['base_character_ids']+row['command_atom_ids']+row['learned_merge_ids']==row['total_ids']*3-9
    save_csv(OUTPUT/'three_metrics.csv',metrics)
    save_csv(OUTPUT/'usage_summary.csv',summaries)
    save_csv(OUTPUT/'per_token_usage.csv',rows)
    flagged=[r for r in rows if r['type']=='learned_merge' and (r['train_count']<=10 or r['train_papers']<=1 or r['bytes']>=16)]
    save_csv(OUTPUT/'review_learned_merges.csv',flagged)
    plot(metrics,summaries,rows)
    print(json.dumps(dict(status='PASS',data_sha256=protocol['data_sha256'],
        sizes=protocol['sizes'],per_token_rows=len(rows),flagged_rows=len(flagged),
        metric_rows=len(metrics),summary_rows=len(summaries)),indent=2))


if __name__=='__main__':main()
