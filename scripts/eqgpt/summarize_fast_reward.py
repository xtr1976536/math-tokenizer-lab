"""Validate and summarize all eight private fast reward-based PDE runs."""
import csv
import json
import re
from pathlib import Path

from math_tokenizer.corpus import ROOT

RUN_ROOTS={
    'Wave_equation':'eqgpt_fast_reward_20261022',
    'Convection_diffusion_equation':'eqgpt_fast_reward_Convection_diffusion_equation_20261022',
    'KG_equation':'eqgpt_fast_reward_KG_equation_20261022',
    'Burgers_equation':'eqgpt_fast_reward_burgers_20261022',
    'Eq_6_2_12':'eqgpt_fast_reward_Eq_6_2_12_20261022',
    'PDE_divide':'eqgpt_fast_reward_PDE_divide_20261022',
    'KdV_equation':'eqgpt_fast_reward_KdV_equation_20261022',
    'Chaffee_Infante_equation':'eqgpt_fast_reward_chaffee_20261022',
}
CASE_LABELS={
    'Wave_equation':'Wave (simple)',
    'Convection_diffusion_equation':'Convection-diffusion (simple)',
    'KG_equation':'Klein-Gordon (medium)',
    'Burgers_equation':'Burgers (medium)',
    'Eq_6_2_12':'Eq. 6.2.12 (medium)',
    'PDE_divide':'PDE divide (medium)',
    'KdV_equation':'KdV (complex)',
    'Chaffee_Infante_equation':'Chafee-Infante (complex)',
}
METHOD_LABELS={'eqgpt_reference':'EqGPT GRE (57)','ast_bpe':'Hard AST-BPE','soft_ast_bpe':'Soft AST-BPE'}


def main():
    report_dir=ROOT/'reports/eqgpt_fast_reward_20261022';report_dir.mkdir(parents=True,exist_ok=True)
    rows=[];case_summaries=[];training_sha=None
    for case,dirname in RUN_ROOTS.items():
        folder=ROOT/'experiments'/dirname
        complete=json.loads((folder/'complete.json').read_text())
        protocol=json.loads((folder/'protocol.json').read_text())
        quality=json.loads((folder/'surrogate_quality.json').read_text())
        assert complete['case']==case and quality['case']==case
        assert len(complete['results'])==7
        assert protocol['seed']==525 and protocol['layers']==3 and protocol['epochs']==25
        assert protocol['attempts_per_cycle']==200 and protocol['cycles']==3
        assert protocol['pinn']=='omitted; no coefficient accuracy measured'
        training_sha=training_sha or protocol['corpus_sha256']
        assert protocol['corpus_sha256']==training_sha
        assert not list(folder.rglob('pinn.json'))
        surrogate_seconds=quality['fit_or_cache_load_seconds']
        validation_match=re.search(r'loss_validate:\s*([0-9.eE+-]+)',quality['loss_last'])
        assert validation_match,quality['loss_last']
        arm_seconds=sum(r['training_seconds']+r['search_seconds'] for r in complete['results'].values())
        case_summaries.append(dict(case=case,label=CASE_LABELS[case],surrogate_fit_seconds=surrogate_seconds,
            seven_arms_train_and_search_seconds=arm_seconds,run_wall_seconds=complete['elapsed_seconds'],
            surrogate_validation_loss=float(validation_match.group(1))))
        for name,result in complete['results'].items():
            if name=='eqgpt_reference':method='eqgpt_reference';size=57
            else:
                method,size=name.rsplit('_',1);size=int(size)
            rows.append(dict(case=case,case_label=CASE_LABELS[case],method=method,
                method_label=METHOD_LABELS[method],total_vocab=size,top1=result['top1'],
                top10_correct=result['top10_correct'],top10_has_redundant=result['top10_redundant'],
                attempted_correct=result['attempted_statuses']['correct'],
                attempted_redundant=result['attempted_statuses']['redundant'],
                attempted_incorrect=result['attempted_statuses']['incorrect'],
                first_correct_attempt=result['first_correct'],training_seconds=result['training_seconds'],
                search_seconds=result['search_seconds'],reward_calls=result['reward_calls'],
                cache_hits=result['cache_hits']))
    with (report_dir/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (report_dir/'summary.json').write_text(json.dumps(dict(cases=case_summaries,arms=rows,
        cases_complete=len(case_summaries),total_arm_runs=len(rows),seed=525,training_corpus_sha256=training_sha),indent=2)+'\n')
    lines=['# Fast reward-based PDE discovery: all eight cases','',
        'This release contains all eight EqGPT benchmark cases. The study uses seed 525, a 3-layer decoder trained for 25 epochs, 200 attempts per cycle for three reward-search cycles, and two continuation epochs per cycle. It compares the author GRE adapter (57 IDs) with Hard and Soft AST-BPE at total vocabularies 280, 315, and 350.',
        '',
        'The author least-squares reward, candidate ranking, Top-10 retention, and continuation training are preserved. The final PINN coefficient-fit stage is omitted; outcomes describe equation structure only. Each case has one author MLP surrogate fitted for 50,000 iterations and shared by all seven arms. All eight target structures are excluded from the shared GPT training corpus. Wave was trained from scratch; its pretrained checkpoints are reused for the other cases.',
        '',
        'Top-1 is the reward-ranked candidate classified by the existing coefficient-free structure matcher. Attempt counts include only candidates successfully parsed and evaluated, so the correct/redundant/incorrect totals can be below 600. One seed supports screening and case-level observations, not a multi-seed success-rate claim.',
        '', '| Case | Method | Total IDs | Top-1 | Top-10 exact | Top-1 extra terms | Correct / redundant / incorrect attempts | First exact attempt |',
        '|---|---|---:|---|---|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case_label']} | {r['method_label']} | {r['total_vocab']} | {r['top1']} | {'yes' if r['top10_correct'] else 'no'} | {'yes' if r['top1']=='redundant' else 'no'} | {r['attempted_correct']} / {r['attempted_redundant']} / {r['attempted_incorrect']} | {r['first_correct_attempt'] if r['first_correct_attempt'] is not None else '-'} |")
    lines+=['','## Runtime','',
        '| Case | Surrogate fit | Seven arm train + search | Surrogate validation loss |','|---|---:|---:|---:|']
    for s in case_summaries:
        lines.append(f"| {s['label']} | {s['surrogate_fit_seconds']/60:.1f} min | {s['seven_arms_train_and_search_seconds']/60:.1f} min | {s['surrogate_validation_loss']:.3g} |")
    lines+=['','All timings use the local Apple MPS runtime. AST-BPE and EqGPT use the same 25-epoch pretraining schedule. For cases after Wave, the pretrained checkpoint was loaded from Wave; the original pretraining duration is retained in each arm summary. Search time includes reward evaluation and continuation training.','',
        '## Per-case artifacts','']
    for case,dirname in RUN_ROOTS.items():
        lines.append(f"- `{case}`: `experiments/{dirname}/` (complete run, logs, source sequences, tokenizer assets, surrogate data/cache, weights and search histories).")
    lines+=['','The complete terminal logs are named `experiments/eqgpt_fast_reward_<case>_20261022.log`. The local original EqGPT data and source are included through the repository’s `third_party/eqgpt/` snapshot. Each case retains its own protocol, source hashes, surrogate quality, and complete JSON result. No PINN output is present.','']
    (report_dir/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='PASS',cases=len(case_summaries),arms=len(rows),report=str(report_dir/'report.md')),indent=2))


if __name__=='__main__':main()
