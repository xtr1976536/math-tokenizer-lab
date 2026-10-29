"""Common-source 80-position corpus, preserving all retained formulas intact.

Start from the previously verified full corpus's common 80-capacity subset.
Retrain each Hard tokenizer only on retained original LaTeX. Recheck every
augmentation and remove any newly oversized source from ALL arms, repeating
until stable. Historical full benchmark and caches are read-only.
"""
import gzip,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.expanded_corpus import source_records,paired_rows,sha
from math_tokenizer.eqgpt.eqgpt_fast_config import ALPHABET,SIZES
from math_tokenizer.eqgpt.eqgpt_fast_tokenizers import train_or_load
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet

OUT=ROOT/'data/formulas/expanded_downstream_80_v1'
BASE=ROOT/'data/formulas/expanded_downstream_v2'
LIMIT=80


def scan(rows,models):
 lengths={'eqgpt_reference':[]};lengths.update({f'ast_bpe_{s}':[] for s in SIZES})
 bad={}
 for row in rows:
  ll={'eqgpt_reference':len(row['augmented_gre_ids'])}
  for (method,size),m in models.items():
   ids=m.encode(row['augmented_latex']);assert m.decode(ids)==row['augmented_latex']
   ll[f'ast_bpe_{size}']=len(ids)+2
  for name,n in ll.items():
   lengths[name].append(n)
   if n>LIMIT:bad.setdefault(row['source_id'],{})[name]=max(n,bad.get(row['source_id'],{}).get(name,0))
 return lengths,bad


def main():
 OUT.mkdir(exist_ok=True);sources=source_records();initial_models={('ast_bpe',s):CompactAlphabet.load(BASE/'tokenizers'/f'ast_bpe_{s}.model.json') for s in SIZES}
 _,bad=scan(paired_rows(sources),initial_models)
 exclusions=[dict(source_id=r['source_id'],source_kind=r['source_kind'],family=r['family'],original_latex=r['original_latex'],reason='full_corpus_models_common_80_limit',over_limit_maxima=bad[r['source_id']]) for r in sources if r['source_id'] in bad]
 keep=[r for r in sources if r['source_id'] not in bad];iterations=[]
 for iteration in range(len(keep)):
  folder=OUT/f'tokenizers_round_{iteration}'
  models=train_or_load([r['original_latex'] for r in keep],folder,SIZES,ALPHABET)
  lengths,oversize=scan(paired_rows(keep),models)
  iterations.append(dict(round=iteration,sources=len(keep),oversize_sources=len(oversize),tokenizer_folder=str(folder.relative_to(OUT))))
  print('LENGTH_CLOSURE',iterations[-1],flush=True)
  if not oversize:break
  for r in keep:
   if r['source_id'] in oversize:exclusions.append(dict(source_id=r['source_id'],source_kind=r['source_kind'],family=r['family'],original_latex=r['original_latex'],reason='subset_tokenizer_round_'+str(iteration),over_limit_maxima=oversize[r['source_id']]))
  keep=[r for r in keep if r['source_id'] not in oversize]
 else:raise RuntimeError('Common subset did not stabilize')
 with (OUT/'sources.jsonl').open('w') as f:
  for row in keep:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 with (OUT/'excluded_length80.jsonl').open('w') as f:
  for row in exclusions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 with gzip.open(OUT/'paired_augmented.jsonl.gz','wt') as f:
  for row in paired_rows(keep):f.write(json.dumps(row,ensure_ascii=False)+'\n')
 stats={name:dict(min=min(v),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=max(v),over_50=sum(n>50 for n in v),over_80=sum(n>80 for n in v)) for name,v in lengths.items()}
 files=['sources.jsonl','paired_augmented.jsonl.gz','excluded_length80.jsonl',*[str(p.relative_to(OUT)) for p in sorted(folder.glob('*.json'))]]
 manifest=dict(version='expanded-downstream-80-v1',source_records=len(keep),augmented_samples=len(keep)*32,
  source_kinds=dict(Counter(r['source_kind'] for r in keep)),original_source_records=len(sources),excluded_source_records=len(exclusions),
  retention_fraction=len(keep)/len(sources),common_filter='all 32 augmentations of a source must fit all arms; remove source from every arm; never truncate',
  training_position_capacity=80,search_max_tokens=80,required_common_positions=max(v['max'] for v in stats.values()),
  tokenizer_folder=str(folder.relative_to(OUT)),tokenizer_corpus='retained unaugmented original LaTeX only; old fixed 98-byte alphabet',
  augmentation='32 sampled equivalent paired variants; original raw LaTeX signed substrings; GRE factors; EOS only last',
  lengths=stats,closure_iterations=iterations,
  input_files={p:sha(ROOT/p) for p in ['data/formulas/gre_benchmark_filtered/benchmark.jsonl','data/formulas/gre_benchmark_audit_v2/eqgpt_221_after_target_exclusion.jsonl']},
  files={p:sha(OUT/p) for p in files},status='PREPARED_PENDING_TWO_PASSES',formal_training_started=False)
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
