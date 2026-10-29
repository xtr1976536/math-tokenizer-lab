"""Prepare faithful paired corpus and raw-LaTeX tokenizers; no Transformer run."""
import gzip,json,time
from pathlib import Path
from collections import Counter
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.expanded_corpus import source_records,paired_rows,sha
from math_tokenizer.eqgpt.eqgpt_fast_config import ALPHABET,SIZES
from math_tokenizer.eqgpt.eqgpt_fast_tokenizers import train_or_load

OUT=ROOT/'data/formulas/expanded_downstream_v2'

def main():
    OUT.mkdir(exist_ok=True);sources=source_records()
    with (OUT/'sources.jsonl').open('w') as f:
        for row in sources:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    count=0
    with gzip.open(OUT/'paired_augmented.jsonl.gz','wt') as f:
        for row in paired_rows(sources):
            f.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
    assert count==72224
    # Only the unaugmented original LaTeX trains the tokenizer. No GRE renderings.
    texts=[r['original_latex'] for r in sources]
    print('CORPUS_READY',len(sources),count,flush=True)
    start=time.perf_counter();models=train_or_load(texts,OUT/'tokenizers',SIZES,ALPHABET)
    lengths={'eqgpt_reference':[]};lengths.update({f'ast_bpe_{s}':[] for s in SIZES})
    with gzip.open(OUT/'paired_augmented.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);lengths['eqgpt_reference'].append(len(row['augmented_gre_ids']))
            for (method,size),m in models.items():
                ids=m.encode(row['augmented_latex']);assert m.decode(ids)==row['augmented_latex']
                lengths[f'ast_bpe_{size}'].append(len(ids)+2)
    import numpy as np
    report={name:dict(min=min(v),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=max(v),over_50=sum(x>50 for x in v)) for name,v in lengths.items()}
    manifest=dict(version='expanded-downstream-v2',source_records=len(sources),augmented_samples=count,
        source_kinds=dict(Counter(r['source_kind'] for r in sources)),
        tokenizer_corpus='2257 original_latex records before augmentation; 98 fixed bytes; capacities 280/315/350',
        augmentation='proposed 32 sampled equivalent permutations; original LaTeX signed substrings preserved; GRE additive/numerator/denominator factors permuted; EOS only last',
        lengths=report,required_common_positions=max(v['max'] for v in report.values()),
        input_files={p:sha(ROOT/p) for p in ['data/formulas/gre_benchmark_filtered/benchmark.jsonl','data/formulas/gre_benchmark_audit_v2/eqgpt_221_after_target_exclusion.jsonl']},
        files={p:sha(OUT/p) for p in ['sources.jsonl','paired_augmented.jsonl.gz',*[str(x.relative_to(OUT)) for x in sorted((OUT/'tokenizers').glob('*.json'))]]},
        preparation_seconds=time.perf_counter()-start,status='CANDIDATE_PENDING_TWO_PASS_AUDIT_AND_USER_SETTINGS',
        formal_training_started=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
