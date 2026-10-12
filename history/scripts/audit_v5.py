"""Independent data/protocol/results audit for v5; no new model selection."""
import json,sys,subprocess
from pathlib import Path
from collections import defaultdict,Counter
from math_tokenizer.corpus import ROOT,canonical
from math_tokenizer.corpus_archive_v5 import ensure_master
ensure_master()
from math_tokenizer.evaluate import read_jsonl,file_hash
from math_tokenizer.benchmark_v5 import template_key,TARGET_CATEGORIES
from math_tokenizer.validation_v2 import validate
from math_tokenizer.rl_vocab_v5 import RLVocab

out=ROOT/sys.argv[1];protocol=json.loads((out/'protocol.json').read_text());selection=json.loads((out/'selection.json').read_text());manifest=json.loads((out/'manifest.json').read_text());data=ROOT/'data/releases/v5/curated';summary=json.loads((data/'summary.json').read_text())
assert file_hash(data/'corpus.jsonl')==summary['corpus_sha256']==protocol['corpus_sha256'];assert file_hash(out/'protocol.json')==manifest['protocol_sha256'];assert file_hash(out/'selection.json')==manifest['selection_sha256'];assert selection['test_used'] is False
splits={name:read_jsonl(data/f'splits/{name}.jsonl') for name in ['train','dev','fresh_paper_test','transfer_train','transfer_dev','extreme_domain_test','synthetic_stress_test']}
for name,digest in protocol['suite_hashes'].items():assert file_hash(data/f'splits/{name}.jsonl')==digest
train= splits['train'];dev=splits['dev'];test=splits['fresh_paper_test']
assert {r['split_group'] for r in train}.isdisjoint(r['split_group'] for r in dev)
assert {r['split_group'] for r in train+dev}.isdisjoint(r['split_group'] for r in test)
exact={canonical(r['latex']) for r in train+dev};templates={template_key(r['latex']) for r in train+dev}
assert all(canonical(r['latex']) not in exact and template_key(r['latex']) not in templates for r in test)
assert all(validate(r['latex']) is None for r in test)
assert {r['id'] for r in splits['extreme_domain_test']}<= {r['id'] for r in test}
sourcecats={'arxiv:'+s['id']:set(s['categories']) for s in json.loads((ROOT/'data/arxiv_sources.json').read_text())+json.loads((ROOT/'data/releases/v5/arxiv_sources.json').read_text())}
for r in splits['transfer_train']+splits['transfer_dev']:
 assert r['domain'] in {'pde','ode','analysis'}
 assert not sourcecats.get(r['source_group'],set())&TARGET_CATEGORIES
new=json.loads((ROOT/'data/releases/v5/arxiv_sources.json').read_text())
for s in new:assert s['license_verified_on_version_page'] and s['version_date'][:4]<'2020' and s['license'].startswith(('CC-BY','CC0'))
metrics=json.loads((out/'metrics.json').read_text());rows=read_jsonl(out/'per_formula.jsonl');by=defaultdict(list)
for r in rows:by[(r['scenario'],r['suite'],r['method'],r['seed'])].append(r)
for m in metrics:
 key=(m['scenario'],m['suite'],m['method'],m['seed']);rr=by[key];assert {r['id'] for r in rr}=={r['id'] for r in splits[m['suite']]};assert len(rr)==len(splits[m['suite']])
 assert sum(r['tokens'] for r in rr)==m['tokens'];assert sum(r['bytes'] for r in rr)==m['bytes'];assert all(r['roundtrip'] for r in rr)
 if m['capacity_budget'] is not None:assert m['vocab_actual']<=m['capacity_budget']
 assert len(m['encode_seconds_repeats'])==5
for name,digest in manifest['models'].items():assert file_hash(out/'models'/name)==digest
# Formal selected RL merges and policy must be present, nonzero, and reward
# corpus IDs entirely within scenario train, never fresh test/held domain data.
replayed=0
for scenario in ('broad','specialized'):
 allowed={r['id'] for r in splits['train' if scenario=='broad' else 'transfer_train']}
 for seed in (17,29,43):
  path=out/'models'/f'{scenario}-rl-selected-{seed}.json';m=RLVocab.load(path);assert m.method=='rl' and any(abs(t)>1e-8 for t in m.theta)
  assert (set(m.fit_ids)|set(m.reward_ids))<=allowed;assert set(m.fit_ids).isdisjoint(m.reward_ids)
  assert any(abs(log['reward']['total']-log['reward']['compression_gain'])>1e-8 for log in m.logs)
  assert all(log['scope']=='training_internal_document_holdout' for log in m.logs)
  rr=[r for r in rows if r['scenario']==scenario and r['method']=='rl_composite' and r['seed']==seed]
  lookup={r['id']:r for name in ('fresh_paper_test','extreme_domain_test','synthetic_stress_test') for r in splits[name]}
  for row in rr:
   text=lookup[row['id']]['latex'];ids=m.encode(text);assert len(ids)==row['tokens'] and m.decode(ids)==text
  replayed+=1
result={'fresh_paper_disjoint':True,'exact_and_template_dedup_vs_fit':True,'extreme_source_category_exclusion':True,'licenses_verified':len(new),
 'same_raw_test_ids_all_models':True,'all_lossless':True,'rl_multiobjective_reward_train_only':True,'selected_rl_replayed':replayed,'model_hashes_verified':len(manifest['models']),
 'runs':len(metrics),'records':len(rows),'limitations':['automatic quality screening not expert mathematical verification','syntax template dedup not semantic equivalence proof','intrinsic OOD is not downstream generalization']}
tests=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT,capture_output=True,text=True);assert tests.returncode==0,tests.stdout+tests.stderr;result['tests']=tests.stdout;(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
